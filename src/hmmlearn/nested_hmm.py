import numpy as np
from scipy.special import softmax, logsumexp
from scipy.optimize import minimize
from sklearn.utils import check_random_state

from .base import _AbstractHMM, ConvergenceMonitor
from .utils import normalize, log_normalize
import time

# NOTE 待办事项（整体优化还是比较麻烦的，先测简化版评分，回头再弄）
## 功能相关
### 1. 目前EM算法的 python 实现涉及多层循环，效率较低，但是正确性已经过保证
### 2. 目前的predict方法正确性未经验证
### 3. 目前没有用到 hmmc.cpp中的logaddexp，可能会有数值稳定性问题，也会影响效率

## 计算速度相关
### 目标是尽量保证一个电视剧的hmm推断能在1小时内完成，否则在多次迭代时，需要考虑切换成cpu资源
### 1. 尝试了用 Cython 优化计算效率，但是 log-lik 计算结果均为 nan，目前无法使用。优先考虑在 Python 中优化
### 2. 以前向算法为例，可以采取以下优化措施（后向算法类似）：
#### a. 存储指定参数的所有人脸转移概率（2^n*2^n）和说话人转移概率（经过归一化，n*n*2^n)，避免反复计算
#### b. 内层上一时刻人脸、说话人状态遍历，改为矩阵运算
#### c. 外层当前时刻人脸，说话人遍历，也可以改为矩阵运算
### 3. 充分统计量计算：也改为矩阵运算


## 内存相关
### 1. 当处理实际电视剧数据时， U, V 矩阵尺寸较大，有可能导致内存不足。如果出现此类问题，可以先将每季的UV统计量存到本地，然后再读取进行累积


class NestedHMM(_AbstractHMM):
    """
    嵌套隐马尔可夫模型
    
    Parameters
    ----------
    n_actors : int
        演员数量
    n_iter : int, optional (default: 100)
        最大迭代次数
    tol : float, optional (default: 1e-2)
        收敛阈值
    verbose : bool, optional (default: False)
        是否打印详细信息
    params : str, optional (default: "abcdefgh")
        控制哪些参数被更新
    init_params : str, optional (default: "abcdefgh")
        控制哪些参数被初始化
    random_state : int or RandomState, optional
        随机种子
    """
    
    def __init__(self, n_actors, n_iter=100, tol=1e-2, verbose=False,
                 params="abcdefgh", init_params="abcdefgh", random_state=None):
        self.n_actors = n_actors    # 演员数量
        self.n_face_states = 2 ** n_actors  # 面部状态数量 (每个演员有2个状态)
        self.n_iter = n_iter    # 最大迭代次数
        self.tol = tol  # 收敛阈值
        self.verbose = verbose  # 是否打印详细信息
        self.params = params    # 控制哪些参数被更新
        self.init_params = init_params  # 控制哪些参数被初始化
        self.random_state = random_state

        # 添加缓存变量，避免反复计算
        ## 预计算的转移矩阵（在每次EM迭代后需要更新）
        self._log_trans_face = np.zeros((self.n_face_states, self.n_face_states))  # [prev_f, f]
        self._log_trans_speaker = np.zeros((self.n_actors, self.n_actors, self.n_face_states))  # [prev_s, s, f]
        ## 所有可能的面部配置
        self.face_configs = self._enumerate_face_configs()

        # 创建监控器
        self.monitor_ = ConvergenceMonitor(tol, n_iter, verbose)

    def _check_and_set_n_features(self, X_1, X_2):
        """
        验证嵌套HMM数据格式，要求
        - X_1: 说话人观测，one-hot编码，形状 (n_samples, n_actors)
        - X_2: 面部出现，二进制数据，形状 (n_samples, n_actors)
        """
        if X_1.shape != X_2.shape:
            raise ValueError(f"X_1 and X_2 must have the same shape, got {X_1.shape} and {X_2.shape}")
        
        if X_1.shape[1] != self.n_actors:
            raise ValueError(f"Expected {self.n_actors} actors, got {X_1.shape}")
            
        # 检查X_1是one-hot编码
        if not np.allclose(X_1.sum(axis=1), 1):
            raise ValueError("X_1 must be one-hot encoded (each row sums to 1)")
        
        # 检查X_2是二进制数据
        if not np.all(np.isin(X_2, [0, 1])):
            raise ValueError("X_2 must contain only binary values (0 or 1)")

    def _validate_lengths(self, X, lengths):
        """
        验证序列长度，要求lengths元素之和等于X的样本数
        """
        if lengths is None:
            return [len(X)]
        
        if np.asarray(lengths).sum() != len(X):
            raise ValueError("Sum of lengths must equal number of samples")
        
        return lengths

    def _enumerate_face_configs(self):
        """
        枚举所有可能的面部配置。返回长为 n_face_states 的列表，每个元素是一个长度为 n_actors 的二进制元组，形如 (0,1,1) 每个位置表示对应演员的人脸是否存在
        """
        face_configs = []
        for i in range(self.n_face_states):
            config = []
            for j in range(self.n_actors):
                config.append((i >> j) & 1)
            face_configs.append(tuple(config))
        return face_configs

    def _init_params(self):
        self.log_likelihood_ = -np.inf
        """初始化嵌套HMM的参数"""
        random_state = check_random_state(self.random_state)
        
        if 'a' in self.init_params:
            # α: 对于每个 actor，其面部出现的初始概率，不要求和为1
            self.alpha_ = random_state.uniform(0.3, 0.7, self.n_actors)
            
        if 'b' in self.init_params:
            # A_F: 面部状态转移矩阵 (n_actors, 2, 2), 每行和为1
            self.A_F_ = np.zeros((self.n_actors, 2, 2))
            for actor in range(self.n_actors):
                for s in range(2):
                    self.A_F_[actor, s] = random_state.dirichlet([2, 1] if s == 0 else [1, 2])

        if 'c' in self.init_params:
            # β: 说话人初始概率的logits,不要求和为1
            self.beta_ = random_state.normal(0, 1, self.n_actors)
            
        if 'd' in self.init_params:
            # γ₁: 面部对说话人初始状态的影响
            self.gamma1_ = random_state.uniform(0.5, 2.0, 1)
            
        if 'e' in self.init_params:
            # A_S: 说话人状态转移矩阵的logits (n_actors, n_actors),不要求和为1
            diag_main = np.diag(random_state.uniform(0.3, 0.7, self.n_actors))
            self.A_S_ = diag_main + (1-diag_main) * random_state.normal(0, 1, (self.n_actors, self.n_actors))
            
        if 'f' in self.init_params:
            # γ₂: 面部对说话人转移的影响
            self.gamma2_ = random_state.uniform(0.5, 2.0, 1)
            
        if 'g' in self.init_params:
            # B_F: 面部识别混淆矩阵 (n_actors, 2, 2), 每行和为1
            self.B_F_ = np.zeros((self.n_actors, 2, 2))
            for actor in range(self.n_actors):
                for s in range(2):
                    self.B_F_[actor, s] = random_state.dirichlet([2, 1] if s == 0 else [1, 2])

        if 'h' in self.init_params:
            # B_S: 说话人识别混淆矩阵 (n_actors, n_actors), 每行和为1
            self.B_S_ = np.zeros((self.n_actors, self.n_actors))
            for actor in range(self.n_actors):
                self.B_S_[actor] = random_state.dirichlet([2 if i == actor else 1 for i in range(self.n_actors)])
        
        self._update_log_transition_matrices()

    def _update_log_transition_matrices(self):
        """预计算用于前向/后向算法的对数转移矩阵。"""
        n_actors = self.n_actors

        # 面部状态转移矩阵: _log_trans_face [prev_f, f]
        for i, prev_config in enumerate(self.face_configs):
            for j, current_config in enumerate(self.face_configs):
                self._log_trans_face[i, j] = self._compute_face_transition_prob(
                    prev_config, current_config)

        # 说话人状态转移张量: _log_trans_speaker[prev_s, s, f]
        for i, config in enumerate(self.face_configs):
            for prev_s in range(n_actors):
                self._log_trans_speaker[prev_s, :, i] = self._compute_speaker_transition_probs(
                    prev_s, config)

    def fit(self, X_1, X_2, lengths=None):
        """训练嵌套HMM模型"""
        X_1 = np.array(X_1)
        X_2 = np.array(X_2)
        
        self._check_and_set_n_features(X_1, X_2)
        lengths = self._validate_lengths(X_1, lengths)
        
        # 初始化参数
        self._init_params()
        # 重置收敛监控器
        self.monitor_._reset()
        
        # EM算法主循环
        for n_iter in range(self.n_iter):
            # E步：计算前向后向概率和期望统计量
            start_time = time.time()
            stats = self._do_estep(X_1, X_2, lengths)
            estep_time = time.time() - start_time

            # 检查收敛
            curr_loglik = stats['log_likelihood'] # 计算当前对数似然
            self.monitor_.report(curr_loglik)
            if self.monitor_.converged:
                break

            # M步：更新参数
            start_time = time.time()
            self._do_mstep(stats, lengths)
            mstep_time = time.time() - start_time

            print(f"E步耗时: {estep_time:.4f}秒")
            print(f"M步耗时: {mstep_time:.4f}秒")

        return self

    def _do_estep(self, X_1, X_2, lengths):
        """E步：使用前向-后向算法计算期望统计量，同时获取数据总体的log-likelihood"""
        
        stats = self._initialize_sufficient_statistics()
        log_likelihood = 0.0
        
        forward_time = 0.0
        backward_time = 0.0
        accumulate_time = 0.0
        
        start_idx = 0
        for length in lengths:
            end_idx = start_idx + length
            
            # 获取当前序列段
            seq_X1 = X_1[start_idx:end_idx]
            seq_X2 = X_2[start_idx:end_idx]
            
            # 前向算法
            start_time = time.time()
            fwd_lattice = self._do_forward_pass(seq_X1, seq_X2)
            forward_time += time.time() - start_time
            
            # 后向算法
            start_time = time.time()
            bwd_lattice = self._do_backward_pass(seq_X1, seq_X2)
            backward_time += time.time() - start_time
            
            # 计算观测序列段的对数似然 $\bbP(\cI_i^{obs}\vert\btheta^{(s)})$
            seq_loglik = logsumexp(fwd_lattice[-1])
            log_likelihood += seq_loglik
            
            # 更新累积统计量，实现对 i=1,...,m 的求和
            start_time = time.time()
            stats_updated = self._accumulate_sufficient_statistics(
                stats, seq_X1, seq_X2, fwd_lattice, bwd_lattice, seq_loglik)
            accumulate_time += time.time() - start_time
            stats = stats_updated

            start_idx = end_idx
            
        stats['log_likelihood'] = log_likelihood
        self.log_likelihood_ = log_likelihood
        
        print(f"前向算法总时间: {forward_time:.4f}秒")
        print(f"后向算法总时间: {backward_time:.4f}秒")
        print(f"累积统计量更新总时间: {accumulate_time:.4f}秒")
        
        return stats

    def _do_forward_pass(self, X_1, X_2):
        """
        前向算法计算前向概率的对数
            - X_1: 说话人观测，形状 (n_samples, n_actors)，one-hot编码
            - X_2: 面部出现，形状 (n_samples, n_actors)，二进制数据
            - return: fwd_lattice, 形状 (n_samples, n_face_states, n_actors), (t, f_idx, s) 表示时刻t面部配置为f_idx，说话人为s的对数概率 $ \log (\bbU_{i,t}(f,\varrho))$
        """
        n_samples = len(X_1)
        n_face_configs = len(self.face_configs)
        
        # fwd_lattice[t, f, s] = log P(观测到t时刻, 面部配置f, 说话人s)
        fwd_lattice = np.full((n_samples, n_face_configs, self.n_actors), -np.inf)
        
        # 初始时刻
        for f_idx, face_config in enumerate(self.face_configs):  # 遍历所有可能的 f
            log_face_prob = self._compute_face_initial_prob(face_config)            
            for speaker in range(self.n_actors):    # 遍历所有可能的\varrho
                log_speaker_prob = self._compute_speaker_initial_prob(speaker, face_config)
                log_emission_prob = self._compute_emission_prob(X_1[0], X_2[0], face_config, speaker)
                
                fwd_lattice[0, f_idx, speaker] = log_face_prob + log_speaker_prob + log_emission_prob
        
        # 递推
        for t in range(1, n_samples):
            ## 检查 fwd_lattice[t-1, :, :] 中是否有 -np.inf
            prev_fwd_lattice = fwd_lattice[t-1, :, :] # shape (n_face_configs, n_actors), corresponds to prev_face_config and prev_speaker
            num_inf = np.sum(prev_fwd_lattice == -np.inf)
            total_elements = prev_fwd_lattice.size
            if num_inf == total_elements:
                continue
            elif num_inf > 0:
                print(f"t={t}: fwd_lattice[t-1] contains {num_inf} -np.inf out of {total_elements} elements")

            ## 计算当前时刻所有可能隐藏状态对应的观测概率
            log_face_emission = list(map(lambda crt_f: self._compute_face_emission_prob(X_2[t], self.face_configs[crt_f]), range(self.n_face_states)))
            log_face_emission = np.array(log_face_emission)  # shape (n_face_configs,), each element corresponds to a current face_config
            log_speaker_emission = np.log(self.B_S_[:, np.argmax(X_1[t])])  # shape (n_actors,), each element corresponds to a current speaker 

            ## 计算当前时刻所有可能的 (f, \varrho)对应的概率log_probs_arr (f_prev, s_prev, f_curr, s_curr)
            log_probs_arr = (prev_fwd_lattice[:, :, None, None] + 
                             self._log_trans_face[:, None, :, None] + np.transpose(self._log_trans_speaker, (0, 2, 1))[None, :, :, :] + 
                             log_face_emission[None, None, :, None] + log_speaker_emission[None, None, None, :])
            ## 对上一时刻的 (f', \varrho') 求和，更新前向概率
            fwd_lattice[t, :, :] = logsumexp(log_probs_arr, axis=(0,1))
        
        return fwd_lattice

    def _do_backward_pass(self, X_1, X_2):
        """
        后向算法计算后向概率的对数
            - X_1: 说话人观测，形状 (n_samples, n_actors)，one-hot编码
            - X_2: 面部出现，形状 (n_samples, n_actors)，二进制数据
            - return: bwd_lattice, 形状 (n_samples, n_face_states, n_actors), (t, f_idx, s) 表示时刻t面部配置为f_idx，说话人为s的对数概率 $ \log (\bbV_{i,t}(f,\varrho))$
        """
        n_samples = len(X_1)
        n_face_configs = len(self.face_configs)
        
        # bwd_lattice[t, f, s] = log P(t+1时刻之后的观测 | t时刻面部配置f, 说话人s)
        bwd_lattice = np.full((n_samples, n_face_configs, self.n_actors), -np.inf)
        
        # 终止时刻
        bwd_lattice[-1, :, :] = 0.0
        
        # 反向递推
        for t in range(n_samples - 2, -1, -1):
            ## 检查 bwd_lattice[t+1, :, :] 中是否有 -np.inf
            next_bwd_lattice = bwd_lattice[t+1, :, :] # shape (n_face_configs, n_actors), corresponds to next_face_config and next_speaker
            num_inf = np.sum(next_bwd_lattice == -np.inf)
            total_elements = next_bwd_lattice.size
            if num_inf == total_elements:
                continue
            elif num_inf > 0:
                print(f"t={t}: bwd_lattice[t+1] contains {num_inf} -np.inf out of {total_elements} elements")

            ## 计算当前时刻所有可能隐藏状态对应的观测概率
            log_face_emission = list(map(lambda next_f: self._compute_face_emission_prob(X_2[t+1], self.face_configs[next_f]), range(self.n_face_states)))
            log_face_emission = np.array(log_face_emission)  # shape (n_face_configs,), each element corresponds to a next_face_config
            log_speaker_emission = np.log(self.B_S_[:, np.argmax(X_1[t+1])])  # shape (n_actors,), each element corresponds to a next_speaker

            ## 计算当前时刻所有可能的 (f, \varrho)对应的概率log_probs_arr (f_curr, s_curr, f_next, s_next)
            log_probs_arr = (next_bwd_lattice[None, None, :, :] + 
                             self._log_trans_face[:, None, :, None] + np.transpose(self._log_trans_speaker, (0, 2, 1))[None, :, :, :] +
                             log_face_emission[None, None, :, None] + log_speaker_emission[None, None, None, :])
            ## 对下一时刻的 (f', \varrho') 求和，更新后向概率
            bwd_lattice[t, :, :] = logsumexp(log_probs_arr, axis=(2,3))
        
        return bwd_lattice

    def _compute_face_initial_prob(self, face_config):
        """
        计算当前面部配置的初始概率 $\bbP(F_{i,1,\cdot}=f)$ 的对数
        """
        log_prob = 0.0
        for actor in range(self.n_actors):
            if face_config[actor] == 1:
                log_prob += np.log(self.alpha_[actor])
            else:
                log_prob += np.log(1 - self.alpha_[actor])
        return log_prob

    def _compute_face_transition_prob(self, prev_config, curr_config):
        """
        计算面部配置的转移概率 $\prod_{\varrho\in\cP} \bbP(F_{i,t,\varrho}\vert F_{i,t-1,\varrho})$ 的对数
        """
        log_prob = sum(list(map(lambda actor: np.log(self.A_F_[actor, prev_config[actor], curr_config[actor]]), range(self.n_actors))))
        return log_prob

    def _compute_speaker_initial_probs(self, face_config):
        """
        计算所有说话人的初始概率 $\bbP(S_{i,1}=\cdot \vert F_{i,1,\cdot}=f)$ 的对数
        """
        logits = np.array([self.beta_[s] + self.gamma1_ * face_config[s] for s in range(self.n_actors)])    # 每个元素代表说话人s的logit
        log_probs = logits - logsumexp(logits)  # of shape (n_actors,), 每个元素代表说话人为s的log概率
        return log_probs

    def _compute_speaker_initial_prob(self, speaker, face_config):
        """
        计算说话人的初始概率 $\bbP(S_{i,1}=\varrho\vert F_{i,1,\cdot}=f)$ 的对数
        """
        log_probs = self._compute_speaker_initial_probs(face_config)
        return log_probs[speaker]

    def _compute_speaker_transition_probs(self, prev_speaker, face_config):
        """
        计算在已知上一时刻说话人时，转移到所有说话人的概率 $\bbP(S_{i,t+1}=\cdot \vert S_{i,t}=\varrho',F_{i,t+1,\cdot}=f)$ 的对数
        """
        logits = np.array(list(map(lambda s: self.A_S_[prev_speaker, s] + self.gamma2_ * face_config[s], range(self.n_actors)))).flatten()
        log_probs = logits - logsumexp(logits)
        return log_probs

    def _compute_speaker_transition_prob(self, prev_speaker, curr_speaker, face_config):
        """
        计算说话人转移概率 $\bbP(S_{i,t+1}=\varrho\vert S_{i,t}=\varrho',F_{i,t+1,\cdot}=f)$ 的对数
        """
        log_probs = self._compute_speaker_transition_probs(prev_speaker, face_config)
        return log_probs[curr_speaker]

    def _compute_face_emission_prob(self, observed_state, face_state):
        """
        计算面部观测概率 $\prod_{\varrho\in\cP} \bB_{\varrho}(F_{i,t,\varrho},\hat F_{i,t,\varrho})$ 的对数
        """
        # 面部观测概率
        log_prob = sum(list(map(lambda actor: np.log(self.B_F_[actor, face_state[actor], observed_state[actor]]), range(self.n_actors))))
        return log_prob

    def _compute_speaker_emission_prob(self, observed_speaker, speaker):
        """
        计算说话人观测概率 $\bB_S(S_{i,t},\hat S_{i,t})$ 的对数
        """
        log_prob = np.log(self.B_S_[speaker, observed_speaker])
        return log_prob

    def _compute_emission_prob(self, x1, x2, face_config, speaker):
        """
        计算发射概率$\bB_S(S_{i,t},\hat S_{i,t}) \prod_{\varrho\in\cP} \bB_{\varrho}(F_{i,t,\varrho},\hat F_{i,t,\varrho})$ 的对数
        """
        # 面部观测概率
        log_prob = self._compute_face_emission_prob(x2, face_config)

        # 说话人观测概率
        speaker_obs = np.argmax(x1)  # one-hot to index
        log_prob += self._compute_speaker_emission_prob(speaker_obs, speaker)
        return log_prob

    def _initialize_sufficient_statistics(self):
        """
        初始化充分统计量，也即 M 步用到的期望值
        """
        return {
            'face_initial_counts': np.zeros(self.n_actors),
            'face_transition_counts': np.zeros((self.n_actors, 2, 2)),
            'speaker_initial_counts': np.zeros((self.n_face_states, self.n_actors)),
            'speaker_transition_counts': np.zeros((self.n_face_states, self.n_actors, self.n_actors)),
            'face_emission_counts': np.zeros((self.n_actors, 2, 2)),
            'speaker_emission_counts': np.zeros((self.n_actors, self.n_actors))
        }

    def _accumulate_sufficient_statistics(self, stats, X_1, X_2, fwd_lattice, bwd_lattice, seq_loglik):
        """
        更新累积充分统计量 stats，以便于后续执行参数更新
        """
        n_samples = len(X_1)
        stats_updated = stats.copy()
        face_configs_arr = np.array(self.face_configs)  # shape (n_face_states, n_actors)
        
        # 计算后验概率
        for t in range(n_samples):
            # 单时刻后验概率 gamma[t, f, s] = P(F_t=f, S_t=s | 全部观测)
            gamma = fwd_lattice[t] + bwd_lattice[t] - seq_loglik
            gamma = np.exp(gamma)   # shape (n_face_states, n_actors)
            gamma_faces = gamma.sum(axis=1)  # shape: (n_face_states,)，提前对speaker求和，方便后续计算面部统计量
            
            # 累积初始统计量
            if t == 0:
                # 计算人脸初始充分统计量 $\bbE\left[\bbN(F_{\cdot,1,\varrho}=1\vert \btheta^{(s)})\right]$ 中属于第i个片段的部分
                ## 先对人脸期望计算式中的 $\varrho'$求和，再对人脸期望计算式中的 $f$ 求和
                stats_updated['face_initial_counts'] += (gamma_faces[:, None] * face_configs_arr).sum(axis=0)
                # 计算说话人初始充分统计量 $\bbE\left[\bbN(F_{\cdot,1,\cdot}=f,S_{\cdot,1}=\varrho\vert \btheta^{(s)})\right] $
                stats_updated['speaker_initial_counts'] += gamma
            
            # 累积转移统计量
            if t > 0:
                ## 计算对数转移后验概率 xi[t-1, f_prev, s_prev, f_curr, s_curr]
                log_face_emission = list(map(lambda crt_f: self._compute_face_emission_prob(X_2[t], self.face_configs[crt_f]), range(self.n_face_states)))
                log_face_emission = np.array(log_face_emission)  # shape (n_face_configs,), each element corresponds to a current face_config
                log_speaker_emission = np.log(self.B_S_[:, np.argmax(X_1[t])])  # shape (n_actors,), each element corresponds to a current speaker 

                log_xi_arr = (fwd_lattice[t-1, :, :, None, None] + self._log_trans_face[:, None, :, None] +
                              np.transpose(self._log_trans_speaker, (0, 2, 1))[None, :, :, :] +
                              log_face_emission[None, None, :, None] + log_speaker_emission[None, None, None, :] +
                              bwd_lattice[t, None, None, :, :] - seq_loglik) 
                xi_arr = np.exp(log_xi_arr) # 求和式中的每一项

                ## 计算面部转移统计量 $\bbE\left[\bbN(F_{\cdot,\cdot-1,\varrho}=\delta,F_{\cdot,\cdot,\varrho}=\delta' \vert \btheta^{(s)})\right]$
                face_transition_weights = xi_arr.sum(axis=(1, 3))  # shape: (n_face_states, n_face_states)
                for actor in range(self.n_actors):  #  人脸期望式中的 $\varrho$
                    prev_states = face_configs_arr[:, actor]  # shape: (n_face_states,), 人脸期望式中的 $\delta$
                    curr_states = face_configs_arr[:, actor]  # shape: (n_face_states,), 人脸期望式中的 $\delta'$
                    for prev_state in [0, 1]:
                        for curr_state in [0, 1]:
                            mask = (prev_states[:, None] == prev_state) & (curr_states[None, :] == curr_state)
                            stats_updated['face_transition_counts'][actor, prev_state, curr_state] += face_transition_weights[mask].sum()
                ## 存储用于说话人转移概率优化的信息
                stats_updated['speaker_transition_counts'] += np.transpose(xi_arr.sum(axis=0), (1, 0, 2))  # sum over prev_f_idx
           
            # 累积发射统计量
            ## 说话人发射统计量
            speaker_obs = np.argmax(X_1[t]) # 说话人期望计算式中的 $\varrho'$
            stats_updated['speaker_emission_counts'][:, speaker_obs] += gamma.sum(axis=0)

            ## 面部发射统计量
            for actor in range(self.n_actors):  # 人脸期望式中的 $\varrho$
                for face_state in [0, 1]:  # 人脸期望式中的 $\delta$
                    mask = (face_configs_arr[:, actor] == face_state)
                    stats_updated['face_emission_counts'][actor, face_state, X_2[t, actor]] += gamma_faces[mask].sum()

        return stats_updated

    def _do_mstep(self, stats, lengths):
        """M步：更新参数"""
        # 更新面部初始概率
        if 'a' in self.params:
            m_segs = len(lengths)
            if m_segs > 0:
                self.alpha_ = stats['face_initial_counts'] / m_segs
                self.alpha_ = np.clip(self.alpha_, 1e-6, 1-1e-6)  # 避免0概率
        
        # 更新面部转移矩阵
        if 'b' in self.params:
            for actor in range(self.n_actors):  # (11)式中的 $\varrho$
                for state in range(2):  # (11)式中的 $\delta$
                    total = stats['face_transition_counts'][actor, state].sum()
                    if total > 0:
                        self.A_F_[actor, state] = stats['face_transition_counts'][actor, state] / total
                        self.A_F_[actor, state] = np.clip(self.A_F_[actor, state], 1e-6, 1-1e-6)
                    else:
                        self.A_F_[actor, state] = np.full(2, 1 / 2)
                        self.A_F_[actor, state, -1] = 1 - self.A_F_[actor, state, :-1].sum()
                        print(f"Warning: Transition probabilities for actor {actor}, state {state} were not updated due to insufficient data. Reset to uniform distribution.")
        
        # 更新说话人初始概率参数 (beta, gamma1)
        if 'c' in self.params or 'd' in self.params:
            self._update_speaker_initial_params(stats)
        
        # 更新说话人转移概率参数 (A_S, gamma2)  
        if 'e' in self.params or 'f' in self.params:
            self._update_speaker_transition_params(stats)
        
        # 更新面部发射矩阵
        if 'g' in self.params:
            for actor in range(self.n_actors):  # (14)式中的 $\varrho$
                for state in range(2):  # (14)式中的 $\delta$
                    total = stats['face_emission_counts'][actor, state].sum()
                    if total > 0:
                        self.B_F_[actor, state] = stats['face_emission_counts'][actor, state] / total # row normalization
                        self.B_F_[actor, state] = np.clip(self.B_F_[actor, state], 1e-6, 1-1e-6)
                    else:
                        self.B_F_[actor, state] = np.full(2, 1 / 2)
                        self.B_F_[actor, state, -1] = 1 - self.B_F_[actor, state, :-1].sum()
                        print(f"Warning: Emission probabilities for actor {actor}, state {state} were not updated due to insufficient data. Reset to uniform distribution.")
        
        # 更新说话人发射矩阵  
        if 'h' in self.params:
            for speaker in range(self.n_actors):  # (15)式中的 $\varrho$
                total = stats['speaker_emission_counts'][speaker].sum()
                if total > 0:
                    self.B_S_[speaker] = stats['speaker_emission_counts'][speaker] / total  # row normalization
                    self.B_S_[speaker] = np.clip(self.B_S_[speaker], 1e-6, 1-1e-6)
                else:
                    self.B_S_[speaker] = np.full(self.n_actors, 1 / self.n_actors)
                    self.B_S_[speaker, -1] = 1 - self.B_S_[speaker, :-1].sum()
                    print(f"Warning: Emission probabilities for speaker {speaker} were not updated due to insufficient data. Reset to uniform distribution.")

        # 更新预计算的对数转移矩阵
        self._update_log_transition_matrices()

    def _update_speaker_initial_params(self, stats):
        """使用数值优化更新说话人初始参数"""
        def objective(params):
            beta, gamma1 = params[:-1], params[-1]
            loss = 0.0

            for f_idx, face_config in enumerate(self.face_configs):  # 说话人期望式中的 $f$
                for speaker in range(self.n_actors):  # 说话人期望式中的 $\varrho$
                    weight = stats['speaker_initial_counts'][f_idx, speaker]
                    if weight > 0:
                        logits = np.array([beta[s] + gamma1 * face_config[s] for s in range(self.n_actors)])  # s 对应 M 步迭代计算式中的 $\varrho'$
                        log_probs = logits - logsumexp(logits)  # log-softmax
                        loss -= weight * log_probs[speaker]
            
            return loss
        
        # 初始参数
        x0 = np.concatenate([self.beta_, self.gamma1_])

            
        # 优化
        result = minimize(objective, x0, method='L-BFGS-B')
        
        if result.success:
            self.beta_ = result.x[:-1]
            self.gamma1_ = np.array([result.x[-1]])
        else:
            print("Warning: Speaker initial parameters optimization did not converge.")

    def _update_speaker_transition_params(self, stats):
        """使用数值优化更新说话人转移参数"""
        def objective(params):
            # 展开A_S矩阵和gamma2
            A_S_flat = params[:-1].reshape(self.n_actors, self.n_actors)
            gamma2 = params[-1]
            
            loss = 0.0

            for prev_speaker in range(self.n_actors):  # 说话人期望式中的 $\varrho$
                for f_idx, face_config in enumerate(self.face_configs):  # 说话人期望计算式中的 $f$
                    for speaker in range(self.n_actors):  # 说话人期望式中的 $\varrho'$
                        weight = stats['speaker_transition_counts'][f_idx, prev_speaker, speaker]
                        if weight > 0:
                            logits = np.array([A_S_flat[prev_speaker, s] + gamma2 * face_config[s] 
                                             for s in range(self.n_actors)])  # s 对应 M 步迭代计算式中的 $\varrho^\ast$
                            log_probs = logits - logsumexp(logits)
                            loss -= weight * log_probs[speaker]
            
            return loss
        
        # 初始参数
        x0 = np.concatenate([self.A_S_.flatten(), self.gamma2_])
        
        # 优化
        result = minimize(objective, x0, method='L-BFGS-B')
        
        if result.success:
            self.A_S_ = result.x[:-1].reshape(self.n_actors, self.n_actors)
            self.gamma2_ = np.array([result.x[-1]])
        else:
            print("Warning: Speaker transition parameters optimization did not converge.")

    def score(self, X_1, X_2, lengths=None):
        """计算观测序列的对数似然"""
        X_1 = np.array(X_1)
        X_2 = np.array(X_2)

        # if self.log_likelihood_ == -np.inf:
        #     raise ValueError("Model must be fitted before scoring.")
        # else:
        #     return self.log_likelihood_            
        return self._do_estep(X_1, X_2, lengths)['log_likelihood']


    def predict_proba(self, X_1, X_2, lengths=None):
        """
        计算给定观测序列时隐藏状态的联合后验概率 $\\bbP(F_{i,t,\cdot}=f,S_{i,t}=\\varrho \\vert \cI_i^{obs}, \\btheta^{(s)})$，以及求和得到的边际后验 $\pi_{i,t,\varrho} = \\bbP(F_{i,t,\\rho}=1 \\vert \cI_i^{obs}, \\btheta^{(s)})$ , $\lambda_{i,t,\\varrho} = \\bbP(S_{i,t}=\\varrho \\vert \cI_i^{obs}, \\btheta^{(s)})$
        
        Parameters
        ----------
        X_1 : array-like, shape (n_samples, n_actors)
            说话人观测，one-hot编码
        X_2 : array-like, shape (n_samples, n_actors)  
            面部出现观测，二进制数据
        lengths : array-like of integers, optional
            每个序列的长度
            
        Returns
        -------
        posteriors : dict
            包含各种后验概率的字典:
            - 'face_states': array, shape (n_samples, n_actors)
              每个时刻每个演员面部出现的后验概率  $ \pi_{i,t,\\varrho} $
            - 'speaker_states': array, shape (n_samples, n_actors)
              每个时刻每个演员是说话人的后验概率  $ \lambda_{i,t,\\varrho} $
            - 'joint_states': array, shape (n_samples, n_face_states, n_actors)
              每个时刻联合状态 (face_config, speaker) 的后验概率 $ \\bbP(F_{i,t,\\cdot}=f,S_{i,t}=\\varrho \\vert \cI_i^{obs}, \\btheta^{(s)}) $
        """
        X_1 = np.array(X_1)
        X_2 = np.array(X_2)
        self._check_and_set_n_features(X_1, X_2)
        lengths = self._validate_lengths(X_1, lengths)
        n_samples = len(X_1)
        face_configs_arr = np.array(self.face_configs)  # shape (n_face_states, n_actors)

        # 初始化输出
        face_posteriors = np.zeros((n_samples, self.n_actors))
        speaker_posteriors = np.zeros((n_samples, self.n_actors))
        joint_posteriors = np.zeros((n_samples, self.n_face_states, self.n_actors))

        # 对每一集的数据        
        start_idx = 0
        for length in lengths:
            end_idx = start_idx + length
            
            ## 获取当前序列段
            seq_X1 = X_1[start_idx:end_idx]
            seq_X2 = X_2[start_idx:end_idx]
            
            ## 计算前向和后向概率，以及序列的对数似然
            fwd_lattice = self._do_forward_pass(seq_X1, seq_X2)
            bwd_lattice = self._do_backward_pass(seq_X1, seq_X2)
            seq_loglik = logsumexp(fwd_lattice[-1])
            
            ## 计算每个时刻的后验概率
            for t in range(length):
                ### 获取并存储联合后验概率 P(F_t=f, S_t=s | 全部观测)
                log_gamma = fwd_lattice[t] + bwd_lattice[t] - seq_loglik
                gamma = np.exp(log_gamma)
                joint_posteriors[start_idx + t] = gamma # of shape (n_face_states, n_actors)
                
                ### 计算面部状态的边际后验概率 P(F_{t, \\rho} =1 | 当前集全部观测)
                for actor in range(self.n_actors):
                    for face_state in [0, 1]:
                        mask = (face_configs_arr[:, actor] == face_state)
                        face_posteriors[start_idx + t, actor] += gamma.sum(axis=1)[mask].sum()  # 先对说话人求和，再对符合要求的面部配置求和
                
                ### 计算说话人状态的边际后验概率 P(S_t=s | 当前集全部观测)
                speaker_posteriors[start_idx + t, :] = gamma.sum(axis=0)
            
            start_idx = end_idx
        
        return {
            'face_states': face_posteriors,
            'speaker_states': speaker_posteriors, 
            'joint_states': joint_posteriors
        }

    def predict(self, X_1, X_2, lengths=None):
        """
        使用Viterbi算法，预测最可能的隐藏状态序列(面部状态和说话人状态)
        
        Parameters
        ----------
        X_1 : array-like, shape (n_samples, n_actors)
            说话人观测，one-hot编码
        X_2 : array-like, shape (n_samples, n_actors)
            面部出现观测，二进制数据
        lengths : array-like of integers, optional
            每个序列的长度，如果为None，则假设是单一序列

        Returns
        -------
        face_states : array, shape (n_samples, n_actors)
            预测的面部状态序列 (0或1)
        speaker_states : array, shape (n_samples,)
            预测的说话人状态序列 (0到n_actors-1)
        """
        X_1 = np.asarray(X_1)
        X_2 = np.asarray(X_2)
        self._check_and_set_n_features(X_1, X_2)
        lengths = self._validate_lengths(X_1, lengths)
        
        # 初始化输出数组
        face_states = np.zeros_like(X_2, dtype=int)
        speaker_states = np.zeros(X_1.shape[0], dtype=int)
        
        start_idx = 0
        for seq_len in lengths:
            end_idx = start_idx + seq_len
            
            # 提取当前序列
            seq_X1 = X_1[start_idx:end_idx]
            seq_X2 = X_2[start_idx:end_idx]
            
            # 使用维特比算法预测
            seq_face_states, seq_speaker_states = self._viterbi(seq_X1, seq_X2)
            
            # 存储结果
            face_states[start_idx:end_idx] = seq_face_states
            speaker_states[start_idx:end_idx] = seq_speaker_states
            
            start_idx = end_idx
            
        return face_states, speaker_states
        
    def _viterbi(self, X_1, X_2):
        """
        对单个序列使用维特比算法进行解码
        
        Parameters
        ----------
        X_1 : array-like, shape (n_frames, n_actors)
            说话人观测序列
        X_2 : array-like, shape (n_frames, n_actors)
            面部出现序列
            
        Returns
        -------
        face_states : array, shape (n_frames, n_actors)
            预测的面部状态序列
        speaker_states : array, shape (n_frames,)
            预测的说话人状态序列
        """
        n_frames = X_1.shape[0]
        
        # 初始化维特比表格 $\delta_{t}(f,s)$ 与回溯路径 $\psi_{t}(f,s)$
        ## $\delta_{t}(f,s)$: 在时刻t，面部配置f，说话人s的最大概率的对数
        viterbi = np.full((n_frames, self.n_face_states, self.n_actors), -np.inf)
        ## $\psi_{t}(f,s)$: t 时刻面部配置f，说话人s时，从1到t的路径中，后验概率最大的路径在 $t-1$ 时刻的状态 (f', s')
        path_face = np.zeros((n_frames, self.n_face_states, self.n_actors), dtype=int)  # 每个元素是 f'在 face_configs 中的索引
        path_speaker = np.zeros((n_frames, self.n_face_states, self.n_actors), dtype=int) # 每个元素是s'的索引
        
        # 初始化: t=0时刻，已知隐状态与观测的联合概率的对数
        for f_idx, face_config in enumerate(self.face_configs):
            for speaker in range(self.n_actors):
                viterbi[0, f_idx, speaker] = self._compute_initial_log_prob(
                    face_config, speaker, X_1[0], X_2[0]
                )
        
        # 前向传播 t=1到n_frames-1
        for t in range(1, n_frames):
            # 计算每个当前状态对应的 $\delta_{t}(f,s)$
            for f_idx, face_config in enumerate(self.face_configs):
                for speaker in range(self.n_actors):
                    max_prob = -np.inf
                    best_prev_f = 0
                    best_prev_s = 0
                    
                    # 遍历所有可能的前一状态
                    for prev_f_idx, prev_face_config in enumerate(self.face_configs):
                        for prev_speaker in range(self.n_actors):
                            if viterbi[t-1, prev_f_idx, prev_speaker] == -np.inf:
                                continue
                            
                            # 计算转移概率
                            trans_log_prob = self._compute_transition_log_prob(
                                prev_face_config, prev_speaker,
                                face_config, speaker,
                                X_1[t], X_2[t]
                            )
                            
                            # 总概率
                            total_prob = viterbi[t-1, prev_f_idx, prev_speaker] + trans_log_prob
                            
                            # 更新最大概率和最佳前一状态
                            if total_prob > max_prob:
                                max_prob = total_prob
                                best_prev_f = prev_f_idx
                                best_prev_s = prev_speaker
                    
                    # 确定 $\delta_{t}(f,s)$
                    viterbi[t, f_idx, speaker] = max_prob
                    # 确定 $\psi_{t}(f,s)$
                    path_face[t, f_idx, speaker] = best_prev_f
                    path_speaker[t, f_idx, speaker] = best_prev_s
        
        # 找到最优路径的结束状态 $i_T^\ast$: best_end_f, best_end_s
        max_prob = -np.inf
        best_end_f = 0
        best_end_s = 0
        for f_idx in range(self.n_face_states):
            for speaker in range(self.n_actors):
                if viterbi[n_frames-1, f_idx, speaker] > max_prob:
                    max_prob = viterbi[n_frames-1, f_idx, speaker]
                    best_end_f = f_idx
                    best_end_s = speaker
        
        # 回溯最优路径
        face_states = np.zeros((n_frames, self.n_actors), dtype=int)
        speaker_states = np.zeros(n_frames, dtype=int)
        curr_f = best_end_f
        curr_s = best_end_s
        for t in range(n_frames-1, -1, -1):
            # 记录当前状态
            face_config = self.face_configs[curr_f]
            for actor in range(self.n_actors):
                face_states[t, actor] = face_config[actor]
            speaker_states[t] = curr_s
            
            # 回溯到前一状态
            if t > 0:
                prev_f = path_face[t, curr_f, curr_s]
                prev_s = path_speaker[t, curr_f, curr_s]
                curr_f = prev_f
                curr_s = prev_s
        
        return face_states, speaker_states

    def _compute_initial_log_prob(self, face_config, speaker, X_1_0, X_2_0):
        """
        计算初始状态的对数概率
        
        Parameters
        ----------
        face_config : array-like, shape (n_actors,)
            面部配置状态
        speaker : int
            说话人状态
        X_1_0 : array-like, shape (n_actors,)
            初始时刻的说话人观测
        X_2_0 : array-like, shape (n_actors,)
            初始时刻的面部观测
            
        Returns
        -------
        log_prob : float
            初始时刻，已知隐状态与观测的对数联合概率
        """
        # 对数初始面部隐藏状态概率
        log_face_init_prob = self._compute_face_initial_prob(face_config)
        # 对数初始说话人隐藏状态概率 P(S_1 | F_1)
        log_speaker_init_prob = self._compute_speaker_initial_prob(speaker, face_config)
        # 对数观测概率 P(F_hat | F)*P(S_hat | S)
        log_obs_prob = self._compute_emission_prob(X_1_0, X_2_0, face_config, speaker)

        log_prob = log_face_init_prob + log_speaker_init_prob + log_obs_prob
        return log_prob
    
    def _compute_transition_log_prob(self, prev_face_config, prev_speaker, 
                                     curr_face_config, curr_speaker, X_1_t, X_2_t):
        """
        计算状态转移的对数概率
        
        Parameters
        ----------
        prev_face_config : array-like, shape (n_actors,)
            前一时刻的面部配置状态
        prev_speaker : int
            前一时刻的说话人状态
        curr_face_config : array-like, shape (n_actors,)
            当前时刻的面部配置状态
        curr_speaker : int
            当前时刻的说话人状态
        X_1_t : array-like, shape (n_actors,)
            当前时刻的说话人观测
        X_2_t : array-like, shape (n_actors,)
            当前时刻的面部观测
            
        Returns
        -------
        log_prob : float
            条件在上一时刻状态下，当前时刻状态的对数概率
        """
        # 对数面部状态转移概率 P(F_t | F_{t-1})
        log_face_trans_prob = self._compute_face_transition_prob(prev_face_config, curr_face_config)
        # 对数说话人状态转移概率 P(S_t | S_{t-1}, F_t)
        log_speaker_trans_prob = self._compute_speaker_transition_prob(prev_speaker, curr_speaker, curr_face_config)
        # 对数观测概率  P(F_hat_t | F_t)*P(S_hat_t | S_t)
        log_obs_prob = self._compute_emission_prob(X_1_t, X_2_t, curr_face_config, curr_speaker)
        
        log_prob = log_face_trans_prob + log_speaker_trans_prob + log_obs_prob
        return log_prob