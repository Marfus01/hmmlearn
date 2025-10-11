import numpy as np
from scipy.special import softmax, logsumexp
from scipy.optimize import minimize
from sklearn.utils import check_random_state

from .base import _AbstractHMM, ConvergenceMonitor
from .utils import normalize, log_normalize

# NOTE
## 1. 目前EM算法完全在 python 中实现，涉及多层循环，效率较低，后续可以考虑用 Cython 优化
## 2. 存储 U, V 矩阵的尺寸较大，有可能导致内存不足
## 3. 目前没有用到 hmmc.cpp中的logaddexp，可能会有数值稳定性问题
## 4. 目前的predict方法效果未经验证


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
    params : str, optional (default: "abcdefg")
        控制哪些参数被更新
    init_params : str, optional (default: "abcdefg")
        控制哪些参数被初始化
    random_state : int or RandomState, optional
        随机种子
    """
    
    def __init__(self, n_actors, n_iter=100, tol=1e-2, verbose=False,
                 params="abcdefg", init_params="abcdefg", random_state=None):
        self.n_actors = n_actors    # 演员数量
        self.n_face_states = 2 ** n_actors  # 面部状态数量 (每个演员有2个状态)
        self.n_iter = n_iter    # 最大迭代次数
        self.tol = tol  # 收敛阈值
        self.verbose = verbose  # 是否打印详细信息
        self.params = params    # 控制哪些参数被更新
        self.init_params = init_params  # 控制哪些参数被初始化
        self.random_state = random_state
        
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
        
        lengths = np.asarray(lengths)
        if lengths.sum() != len(X):
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

            # B_S: 说话人识别混淆矩阵 (n_actors, n_actors), 每行和为1
            self.B_S_ = np.zeros((self.n_actors, self.n_actors))
            for actor in range(self.n_actors):
                self.B_S_[actor] = random_state.dirichlet([2 if i == actor else 1 for i in range(self.n_actors)])
 
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
            stats = self._do_estep(X_1, X_2, lengths)

            # 检查收敛
            curr_loglik = stats['log_likelihood'] # 计算当前对数似然
            self.monitor_.history.append(curr_loglik)
            self.monitor_.iter = n_iter
            self.monitor_.report(curr_loglik)
            if self.monitor_.converged:
                break
        
            # M步：更新参数
            self._do_mstep(stats, lengths)

        return self

    def _do_estep(self, X_1, X_2, lengths):
        """E步：使用前向-后向算法计算期望统计量，同时获取数据总体的log-likelihood"""
        stats = self._initialize_sufficient_statistics()
        log_likelihood = 0.0
        
        start_idx = 0
        for length in lengths:
            end_idx = start_idx + length
            
            # 获取当前序列段
            seq_X1 = X_1[start_idx:end_idx]
            seq_X2 = X_2[start_idx:end_idx]
            
            # 前向-后向算法
            fwd_lattice = self._do_forward_pass(seq_X1, seq_X2)
            bwd_lattice = self._do_backward_pass(seq_X1, seq_X2)
            
            # 计算观测序列段的对数似然 $\bbP(\cI_i^{obs}\vert\btheta^{(s)})$
            seq_loglik = logsumexp(fwd_lattice[-1])
            log_likelihood += seq_loglik
            
            # 更新累积统计量，实现对 i=1,...,m 的求和
            stats_updated = self._accumulate_sufficient_statistics(
                stats, seq_X1, seq_X2, fwd_lattice, bwd_lattice, seq_loglik)
            stats = stats_updated

            start_idx = end_idx
            
        stats['log_likelihood'] = log_likelihood
        return stats

    def _do_forward_pass(self, X_1, X_2):
        """
        前向算法计算前向概率的对数
            - X_1: 说话人观测，形状 (n_samples, n_actors)，one-hot编码
            - X_2: 面部出现，形状 (n_samples, n_actors)，二进制数据
            - return: fwd_lattice, 形状 (n_samples, n_face_states, n_actors), (t, f_idx, s) 表示时刻t面部配置为f_idx，说话人为s的对数概率 $ \log (\bbU_{i,t}(f,\varrho))$
        """
        n_samples = len(X_1)
        face_configs = self._enumerate_face_configs()
        n_face_configs = len(face_configs)
        
        # fwd_lattice[t, f, s] = log P(观测到t时刻, 面部配置f, 说话人s)
        fwd_lattice = np.full((n_samples, n_face_configs, self.n_actors), -np.inf)
        
        # 初始时刻
        for f_idx, face_config in enumerate(face_configs):  # 遍历所有可能的 f
            log_face_prob = self._compute_face_initial_prob(face_config)            
            for speaker in range(self.n_actors):    # 遍历所有可能的\varrho
                log_speaker_prob = self._compute_speaker_initial_prob(speaker, face_config)
                log_emission_prob = self._compute_emission_prob(X_1[0], X_2[0], face_config, speaker)
                
                fwd_lattice[0, f_idx, speaker] = log_face_prob + log_speaker_prob + log_emission_prob
        
        # 递推
        for t in range(1, n_samples):
            ## 当前时刻所有可能的 (f, \varrho)
            for f_idx, face_config in enumerate(face_configs):
                for speaker in range(self.n_actors):
                    log_probs = []

                    ## 前一时刻所有可能的 (f', \varrho')
                    for prev_f_idx, prev_face_config in enumerate(face_configs):
                        for prev_speaker in range(self.n_actors):
                            # NOTE: 后面可以打印调试，看看这里是否有大量的 -inf
                            if fwd_lattice[t-1, prev_f_idx, prev_speaker] == -np.inf:   # 如果前一时刻的 U 为负无穷，则跳过
                                continue
                                
                            # 面部转移概率
                            log_face_trans = self._compute_face_transition_prob(prev_face_config, face_config)
                            # 说话人转移概率
                            log_speaker_trans = self._compute_speaker_transition_prob(prev_speaker, speaker, face_config)
                            # 发射概率
                            log_emission = self._compute_emission_prob(X_1[t], X_2[t], face_config, speaker)
                            
                            # 计算迭代公式里被求和的每一项的对数
                            log_prob = (fwd_lattice[t-1, prev_f_idx, prev_speaker] + 
                                      log_face_trans + log_speaker_trans + log_emission)
                            log_probs.append(log_prob)
                    
                    if log_probs:
                        fwd_lattice[t, f_idx, speaker] = logsumexp(log_probs)   # 当前时刻的 U
        
        return fwd_lattice

    def _do_backward_pass(self, X_1, X_2):
        """
        后向算法计算后向概率的对数
            - X_1: 说话人观测，形状 (n_samples, n_actors)，one-hot编码
            - X_2: 面部出现，形状 (n_samples, n_actors)，二进制数据
            - return: bwd_lattice, 形状 (n_samples, n_face_states, n_actors), (t, f_idx, s) 表示时刻t面部配置为f_idx，说话人为s的对数概率 $ \log (\bbV_{i,t}(f,\varrho))$
        """
        n_samples = len(X_1)
        face_configs = self._enumerate_face_configs()
        n_face_configs = len(face_configs)
        
        # bwd_lattice[t, f, s] = log P(t+1时刻之后的观测 | t时刻面部配置f, 说话人s)
        bwd_lattice = np.full((n_samples, n_face_configs, self.n_actors), -np.inf)
        
        # 终止时刻
        bwd_lattice[-1, :, :] = 0.0
        
        # 反向递推
        for t in range(n_samples - 2, -1, -1):
            ## 当前时刻所有可能的 (f, \varrho)
            for f_idx, face_config in enumerate(face_configs):
                for speaker in range(self.n_actors):
                    log_probs = []
                    
                    ## 下一时刻所有可能的 (f', \varrho')
                    for next_f_idx, next_face_config in enumerate(face_configs):
                        for next_speaker in range(self.n_actors):
                            # NOTE: 后面可以打印调试，看看这里是否有大量的 -inf
                            if bwd_lattice[t+1, next_f_idx, next_speaker] == -np.inf:
                                continue
                                
                            # 面部转移概率
                            log_face_trans = self._compute_face_transition_prob(face_config, next_face_config)
                            # 说话人转移概率  
                            log_speaker_trans = self._compute_speaker_transition_prob(speaker, next_speaker, next_face_config)
                            # 发射概率
                            log_emission = self._compute_emission_prob(X_1[t+1], X_2[t+1], next_face_config, next_speaker)
                            
                            log_prob = (log_face_trans + log_speaker_trans + log_emission + 
                                      bwd_lattice[t+1, next_f_idx, next_speaker])
                            log_probs.append(log_prob)
                    
                    if log_probs:
                        bwd_lattice[t, f_idx, speaker] = logsumexp(log_probs)
        
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
        log_prob = 0.0
        for actor in range(self.n_actors):
            prev_state = prev_config[actor]
            curr_state = curr_config[actor]
            log_prob += np.log(self.A_F_[actor, prev_state, curr_state])
        return log_prob

    def _compute_speaker_initial_prob(self, speaker, face_config):
        """
        计算说话人的初始概率 $\bbP(S_{i,1}=\varrho\vert F_{i,1,\cdot}=f)$ 的对数
        """
        # NOTE: 这里算出了所有说话人的概率，但是却只返回了 speaker 对应的概率，后续可以优化
        logits = np.array([self.beta_[s] + self.gamma1_ * face_config[s] for s in range(self.n_actors)])    # 每个元素代表说话人s的logit
        log_probs = logits - logsumexp(logits)  # of shape (n_actors,), 每个元素代表说话人为s的log概率
        return log_probs[speaker]

    def _compute_speaker_transition_prob(self, prev_speaker, curr_speaker, face_config):
        """
        计算说话人转移概率 $\bbP(S_{i,t+1}=\varrho\vert S_{i,t}=\varrho',F_{i,t+1,\cdot}=f)$ 的对数
        """
        # NOTE: 这里算出了所有说话人的概率，但是却只返回了 speaker 对应的概率，后续可以优化
        logits = np.array([self.A_S_[prev_speaker, s] + self.gamma2_ * face_config[s] 
                          for s in range(self.n_actors)])
        log_probs = logits - logsumexp(logits)
        return log_probs[curr_speaker]

    def _compute_emission_prob(self, x1, x2, face_config, speaker):
        """
        计算发射概率$\bB_S(S_{i,t},\hat S_{i,t}) \prod_{\varrho\in\cP} \bB_{\varrho}(F_{i,t,\varrho},\hat F_{i,t,\varrho})$ 的对数
        """
        log_prob = 0.0
        
        # 说话人观测概率
        speaker_obs = np.argmax(x1)  # one-hot to index
        log_prob += np.log(self.B_S_[speaker, speaker_obs])
        
        # 面部观测概率
        for actor in range(self.n_actors):
            true_face = face_config[actor]
            obs_face = x2[actor]
            log_prob += np.log(self.B_F_[actor, true_face, obs_face])
            
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
        face_configs = self._enumerate_face_configs()
        stats_updated = stats.copy()
        
        # 计算后验概率
        for t in range(n_samples):
            # 单时刻后验概率 gamma[t, f, s] = P(F_t=f, S_t=s | 全部观测)
            gamma = fwd_lattice[t] + bwd_lattice[t] - seq_loglik
            gamma = np.exp(gamma)
            
            # 累积初始统计量
            if t == 0:
                for f_idx, face_config in enumerate(face_configs):  # 人脸期望计算式中的 $f$，说话人期望式中的 $f$
                    for speaker in range(self.n_actors):  # 人脸期望计算式中的 $\varrho'$，说话人期望式中的 $\varrho$
                        weight = gamma[f_idx, speaker]
                        # NOTE: 计算效率待优化，可以仅对 face_config 中为 1 的位置做 for 循环
                        # 计算人脸初始充分统计量 $\bbE\left[\bbN(F_{\cdot,1,\varrho}=1\vert \btheta^{(s)})\right]$ 中属于第i个片段的部分
                        for actor in range(self.n_actors):  # 人脸期望式中的 $\varrho$
                            if face_config[actor] == 1:
                                stats_updated['face_initial_counts'][actor] += weight
                        
                        # 计算说话人初始充分统计量 $\bbE\left[\bbN(F_{\cdot,1,\cdot}=f,S_{\cdot,1}=\varrho\vert \btheta^{(s)})\right] $
                        stats_updated['speaker_initial_counts'][f_idx, speaker] += weight
            
            # 累积转移统计量
            if t > 0:
                # 计算转移后验概率 xi[t-1, f_prev, s_prev, f_curr, s_curr]
                for prev_f_idx, prev_face_config in enumerate(face_configs):  # 人脸期望计算式中的 $f$，说话人期望计算式中的 $f'$
                    for prev_speaker in range(self.n_actors):  # 人脸期望计算式中的 $\varrho'$，说话人期望式中的 $\varrho$
                        for f_idx, face_config in enumerate(face_configs):  # 人脸期望计算式中的 $f'$，说话人期望式中的 $f$
                            for speaker in range(self.n_actors):  # 人脸期望计算式中的 $\varrho^\ast$，说话人期望式中的 $\varrho'$
                                
                                log_xi = (fwd_lattice[t-1, prev_f_idx, prev_speaker] +
                                         self._compute_face_transition_prob(prev_face_config, face_config) +
                                         self._compute_speaker_transition_prob(prev_speaker, speaker, face_config) +
                                         self._compute_emission_prob(X_1[t], X_2[t], face_config, speaker) +
                                         bwd_lattice[t, f_idx, speaker] - seq_loglik)
                                
                                xi = np.exp(log_xi) # 求和式中的每一项
                                
                                # 计算面部转移统计量 $\bbE\left[\bbN(F_{\cdot,\cdot-1,\varrho}=\delta,F_{\cdot,\cdot,\varrho}=\delta' \vert \btheta^{(s)})\right]$
                                for actor in range(self.n_actors):  #  人脸期望式中的 $\varrho$
                                    prev_state = prev_face_config[actor]  # 人脸期望式中的 $\delta$
                                    curr_state = face_config[actor] # 人脸期望式中的 $\delta'$
                                    stats_updated['face_transition_counts'][actor, prev_state, curr_state] += xi
                                
                                # 存储用于说话人转移概率优化的信息
                                # 与说话人初始概率类似，需要同时保存 (f, \varrho, \varrho')的信息和计算式中三层求和式内项的取值
                                stats_updated['speaker_transition_counts'][f_idx, prev_speaker, speaker] += xi
            
            # 累积发射统计量
            speaker_obs = np.argmax(X_1[t]) # 说话人期望计算式中的 $\varrho'$
            for f_idx, face_config in enumerate(face_configs):  # 期望计算式中的 $f$
                for speaker in range(self.n_actors):
                    weight = gamma[f_idx, speaker]  # 人脸期望计算式中的 $\varrho$，说话人期望计算式中的 $\varrho'$

                    # 面部发射统计量
                    for actor in range(self.n_actors):  # 人脸期望式中的 $\varrho$
                        true_face = face_config[actor]  # 人脸期望式中的 $\delta$
                        obs_face = X_2[t, actor]  # 人脸期望式中的 $\delta'$
                        stats_updated['face_emission_counts'][actor, true_face, obs_face] += weight

                    # 说话人发射统计量
                    stats_updated['speaker_emission_counts'][speaker, speaker_obs] += weight
        
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
            for actor in range(self.n_actors):
                for state in range(2):
                    total = stats['face_transition_counts'][actor, state].sum()
                    if total > 0:
                        self.A_F_[actor, state] = stats['face_transition_counts'][actor, state] / total
                        self.A_F_[actor, state] = np.clip(self.A_F_[actor, state], 1e-6, 1-1e-6)
        
        # 更新说话人初始概率参数 (beta, gamma1)
        if 'c' in self.params or 'd' in self.params:
            self._update_speaker_initial_params(stats)
        
        # 更新说话人转移概率参数 (A_S, gamma2)  
        if 'e' in self.params or 'f' in self.params:
            self._update_speaker_transition_params(stats)
        
        # 更新面部发射矩阵
        if 'f' in self.params:  # 重用f参数位置
            for actor in range(self.n_actors):
                for state in range(2):
                    total = stats['face_emission_counts'][actor, state].sum()
                    if total > 0:
                        self.B_F_[actor, state] = stats['face_emission_counts'][actor, state] / total
                        self.B_F_[actor, state] = np.clip(self.B_F_[actor, state], 1e-6, 1-1e-6)
        
        # 更新说话人发射矩阵  
        if 'g' in self.params:
            for speaker in range(self.n_actors):
                total = stats['speaker_emission_counts'][speaker].sum()
                if total > 0:
                    self.B_S_[speaker] = stats['speaker_emission_counts'][speaker] / total
                    self.B_S_[speaker] = np.clip(self.B_S_[speaker], 1e-6, 1-1e-6)

    def _update_speaker_initial_params(self, stats):
        """使用数值优化更新说话人初始参数"""
        def objective(params):
            beta, gamma1 = params[:-1], params[-1]
            loss = 0.0
            face_configs = self._enumerate_face_configs()

            for f_idx, face_config in enumerate(face_configs):  # 说话人期望式中的 $f$
                for speaker in range(self.n_actors):  # 说话人期望式中的 $\varrho$
                    weight = stats['speaker_initial_counts'][f_idx, speaker]
                    if weight > 0:
                        logits = np.array([beta[s] + gamma1 * face_config[s] for s in range(self.n_actors)])  # s 对应 M 步迭代计算式中的 $\varrho'$
                        log_probs = logits - logsumexp(logits)  # log-softmax
                        loss -= weight * log_probs[speaker]
            
            return loss
        
        # 初始参数
        x0 = np.concatenate([self.beta_, [self.gamma1_]])
        
        # 优化
        result = minimize(objective, x0, method='L-BFGS-B')
        
        if result.success:
            self.beta_ = result.x[:-1]
            self.gamma1_ = result.x[-1]

    def _update_speaker_transition_params(self, stats):
        """使用数值优化更新说话人转移参数"""
        def objective(params):
            # 展开A_S矩阵和gamma2
            A_S_flat = params[:-1].reshape(self.n_actors, self.n_actors)
            gamma2 = params[-1]
            
            loss = 0.0
            face_configs = self._enumerate_face_configs()

            for prev_speaker in range(self.n_actors):  # 说话人期望式中的 $\varrho$
                for f_idx, face_config in enumerate(face_configs):  # 说话人期望计算式中的 $f$
                    for speaker in range(self.n_actors):  # 说话人期望式中的 $\varrho'$
                        weight = stats['speaker_transition_counts'][f_idx, prev_speaker, speaker]
                        if weight > 0:
                            logits = np.array([A_S_flat[prev_speaker, s] + gamma2 * face_config[s] 
                                             for s in range(self.n_actors)])  # s 对应 M 步迭代计算式中的 $\varrho^\ast$
                            log_probs = logits - logsumexp(logits)
                            loss -= weight * log_probs[speaker]
            
            return loss
        
        # 初始参数
        x0 = np.concatenate([self.A_S_.flatten(), [self.gamma2_]])
        
        # 优化
        result = minimize(objective, x0, method='L-BFGS-B')
        
        if result.success:
            self.A_S_ = result.x[:-1].reshape(self.n_actors, self.n_actors)
            self.gamma2_ = result.x[-1]

    def score(self, X_1, X_2, lengths=None):
        """计算观测序列的对数似然"""
        X_1 = np.array(X_1)
        X_2 = np.array(X_2)

        return self._do_estep(X_1, X_2, lengths)['log_likelihood']

    def predict(self, X_2, lengths=None):
        """根据面部观测预测说话人序列"""
        if lengths is None:
            lengths = [len(X_2)]
        
        face_configs = self._enumerate_face_configs()
        predictions = []
        
        start_idx = 0
        for length in lengths:
            end_idx = start_idx + length
            F_hat_seq = X_2[start_idx:end_idx]
            
            # 使用Viterbi算法找到最可能的隐状态序列
            best_path = self._viterbi(F_hat_seq, face_configs)
            predictions.extend(best_path)
            
            start_idx = end_idx
        
        return np.array(predictions)

    def _viterbi(self, F_hat_seq, face_configs):
        """Viterbi算法解码最可能的说话人序列"""
        T = len(F_hat_seq)
        n_face_configs = len(face_configs)
        
        # 初始化
        log_prob = np.full((T, n_face_configs, self.n_actors), -np.inf)
        path = np.zeros((T, n_face_configs, self.n_actors), dtype=int)
        
        # t=0
        for f_idx, f in enumerate(face_configs):
            for speaker in range(self.n_actors):
                log_prob[0, f_idx, speaker] = self._compute_initial_log_prob(f, speaker, F_hat_seq[0])
        
        # 递推
        for t in range(1, T):
            for f_idx, f in enumerate(face_configs):
                for speaker in range(self.n_actors):
                    best_prob = -np.inf
                    best_prev = 0
                    
                    for prev_f_idx, prev_f in enumerate(face_configs):
                        for prev_speaker in range(self.n_actors):
                            trans_prob = self._compute_transition_log_prob(
                                prev_f, f, prev_speaker, speaker, F_hat_seq[t])
                            
                            prob = log_prob[t-1, prev_f_idx, prev_speaker] + trans_prob
                            if prob > best_prob:
                                best_prob = prob
                                best_prev = prev_f_idx * self.n_actors + prev_speaker
                    
                    log_prob[t, f_idx, speaker] = best_prob
                    path[t, f_idx, speaker] = best_prev
        
        # 回溯
        best_speakers = np.zeros(T, dtype=int)
        best_final = np.unravel_index(np.argmax(log_prob[T-1]), log_prob[T-1].shape)
        
        current_f_idx, current_speaker = best_final
        best_speakers[T-1] = current_speaker
        
        for t in range(T-2, -1, -1):
            prev_state = path[t+1, current_f_idx, current_speaker]
            current_f_idx = prev_state // self.n_actors
            current_speaker = prev_state % self.n_actors
            best_speakers[t] = current_speaker
        
        return best_speakers

    def _compute_initial_log_prob(self, f, speaker, f_hat):
        """计算初始状态的对数概率"""
        log_prob = 0.0
        
        # P(F₁)
        for actor in range(self.n_actors):
            log_prob += np.log(self.alpha_[actor] if f[actor] else 1 - self.alpha_[actor])
        
        # P(S₁|F₁)
        logits = self.beta_ + self.gamma1_ * np.array(f)
        log_prob += logits[speaker] - logsumexp(logits)
        
        # P(F̂₁|F₁) * P(Ŝ₁|S₁) - 这里简化处理观测
        for actor in range(self.n_actors):
            log_prob += np.log(self.B_F_[actor, f[actor], f_hat[actor]])
        
        return log_prob

    def _compute_transition_log_prob(self, prev_f, f, prev_speaker, speaker, f_hat):
        """计算状态转移的对数概率"""
        log_prob = 0.0
        
        # P(Fₜ|Fₜ₋₁)
        for actor in range(self.n_actors):
            log_prob += np.log(self.A_F_[actor, prev_f[actor], f[actor]])
        
        # P(Sₜ|Sₜ₋₁, Fₜ)
        logits = self.A_S_[prev_speaker] + self.gamma2_ * np.array(f)
        log_prob += logits[speaker] - logsumexp(logits)
        
        # P(F̂ₜ|Fₜ)
        for actor in range(self.n_actors):
            log_prob += np.log(self.B_F_[actor, f[actor], f_hat[actor]])
        
        return log_prob