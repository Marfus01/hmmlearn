import numpy as np
from scipy.special import softmax, logsumexp
from scipy.optimize import minimize
from sklearn.utils import check_random_state

from .base import _AbstractHMM, ConvergenceMonitor
from .utils import normalize, log_normalize

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
        self.n_actors = n_actors
        self.n_face_states = 2 ** n_actors
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose
        self.params = params
        self.init_params = init_params
        self.random_state = random_state
        
        # 创建监控器
        self.monitor_ = ConvergenceMonitor(tol, n_iter, verbose)

    def _check_and_set_n_features(self, X_1, X_2):
        """验证嵌套HMM数据格式并设置特征数"""
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
        """验证序列长度"""
        if lengths is None:
            return [len(X)]
        
        lengths = np.asarray(lengths)
        if lengths.sum() != len(X):
            raise ValueError("Sum of lengths must equal number of samples")
        
        return lengths

    def _enumerate_face_configs(self):
        """枚举所有可能的面部配置"""
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
            # α: 面部出现的初始概率
            self.alpha_ = random_state.uniform(0.3, 0.7, self.n_actors)
            
        if 'b' in self.init_params:
            # A_F: 面部状态转移矩阵 (n_actors, 2, 2)
            self.A_F_ = np.zeros((self.n_actors, 2, 2))
            for actor in range(self.n_actors):
                for s in range(2):
                    self.A_F_[actor, s] = random_state.dirichlet([1, 1])
                
        if 'c' in self.init_params:
            # β: 说话人初始概率的logits
            self.beta_ = random_state.normal(0, 1, self.n_actors)
            
        if 'd' in self.init_params:
            # γ₁: 面部对说话人初始状态的影响
            self.gamma1_ = random_state.uniform(0.5, 2.0)
            
        if 'e' in self.init_params:
            # A_S: 说话人状态转移矩阵的logits (n_actors, n_actors)
            self.A_S_ = random_state.normal(0, 1, (self.n_actors, self.n_actors))
            
        if 'f' in self.init_params:
            # γ₂: 面部对说话人转移的影响
            self.gamma2_ = random_state.uniform(0.5, 2.0)
            
        if 'g' in self.init_params:
            # B_F: 面部识别混淆矩阵 (n_actors, 2, 2)
            self.B_F_ = np.zeros((self.n_actors, 2, 2))
            for actor in range(self.n_actors):
                for s in range(2):
                    self.B_F_[actor, s] = random_state.dirichlet([2, 1] if s == 0 else [1, 2])
            
            # B_S: 说话人识别混淆矩阵 (n_actors, n_actors)
            self.B_S_ = np.zeros((self.n_actors, self.n_actors))
            for actor in range(self.n_actors):
                self.B_S_[actor] = random_state.dirichlet([2 if i == actor else 1 for i in range(self.n_actors)])

    def _compute_log_likelihood(self, X_1, X_2, lengths):
        """计算对数似然"""
        # 将X_1转换为标签格式
        S_hat = np.argmax(X_1, axis=1)
        
        # 枚举所有面部配置
        face_configs = self._enumerate_face_configs()
        
        log_prob = 0.0
        start_idx = 0
        
        for length in lengths:
            end_idx = start_idx + length
            S_hat_seq = S_hat[start_idx:end_idx]
            F_hat_seq = X_2[start_idx:end_idx]
            
            # 前向算法计算该序列的似然
            forward_probs = self._forward(F_hat_seq, S_hat_seq, face_configs)
            seq_log_prob = logsumexp(forward_probs[-1].flatten())
            log_prob += seq_log_prob
            
            start_idx = end_idx
            
        return log_prob

    def _forward(self, F_hat_seq, S_hat_seq, face_configs):
        """前向算法"""
        T = len(F_hat_seq)
        n_states = len(face_configs) * self.n_actors
        
        # 初始化前向概率矩阵
        forward_probs = np.zeros((T, len(face_configs), self.n_actors))
        
        # t=1的初始化
        for f_idx, f in enumerate(face_configs):
            for speaker in range(self.n_actors):
                # P(F₁) * P(S₁|F₁) * P(F̂₁|F₁) * P(Ŝ₁|S₁)
                log_prob = 0.0
                
                # P(F₁) = ∏ᵨ Bernoulli(F₁ᵨ|αᵨ)
                for actor in range(self.n_actors):
                    if f[actor] == 1:
                        log_prob += np.log(self.alpha_[actor])
                    else:
                        log_prob += np.log(1 - self.alpha_[actor])
                
                # P(S₁|F₁) = Multinomial(S₁|β_ω₁)
                logits = self.beta_ + self.gamma1_ * np.array(f)
                log_prob += logits[speaker] - logsumexp(logits)
                
                # P(F̂₁|F₁) = ∏ᵨ B_F[ᵨ](F₁ᵨ, F̂₁ᵨ)
                for actor in range(self.n_actors):
                    log_prob += np.log(self.B_F_[actor, f[actor], F_hat_seq[0, actor]])
                
                # P(Ŝ₁|S₁) = B_S(S₁, Ŝ₁)
                log_prob += np.log(self.B_S_[speaker, S_hat_seq[0]])
                
                forward_probs[0, f_idx, speaker] = log_prob
        
        # 递推计算t=2,...,T
        for t in range(1, T):
            for f_idx, f in enumerate(face_configs):
                for speaker in range(self.n_actors):
                    log_prob = -np.inf
                    
                    for prev_f_idx, prev_f in enumerate(face_configs):
                        for prev_speaker in range(self.n_actors):
                            # P(Fₜ|Fₜ₋₁)
                            trans_log_prob = 0.0
                            for actor in range(self.n_actors):
                                trans_log_prob += np.log(self.A_F_[actor, prev_f[actor], f[actor]])
                            
                            # P(Sₜ|Sₜ₋₁, Fₜ)
                            logits = self.A_S_[prev_speaker] + self.gamma2_ * np.array(f)
                            trans_log_prob += logits[speaker] - logsumexp(logits)
                            
                            # P(F̂ₜ|Fₜ)
                            for actor in range(self.n_actors):
                                trans_log_prob += np.log(self.B_F_[actor, f[actor], F_hat_seq[t, actor]])
                            
                            # P(Ŝₜ|Sₜ)
                            trans_log_prob += np.log(self.B_S_[speaker, S_hat_seq[t]])
                            
                            # 累加前向概率
                            log_prob = np.logaddexp(log_prob, 
                                                  forward_probs[t-1, prev_f_idx, prev_speaker] + trans_log_prob)
                    
                    forward_probs[t, f_idx, speaker] = log_prob
        
        return forward_probs

    def _backward(self, F_hat_seq, S_hat_seq, face_configs):
        """后向算法"""
        T = len(F_hat_seq)
        backward_probs = np.zeros((T, len(face_configs), self.n_actors))
        
        # 初始化最后一个时间步
        backward_probs[T-1] = 0.0  # log(1) = 0
        
        # 递推计算t=T-1,...,1
        for t in range(T-2, -1, -1):
            for f_idx, f in enumerate(face_configs):
                for speaker in range(self.n_actors):
                    log_prob = -np.inf
                    
                    for next_f_idx, next_f in enumerate(face_configs):
                        for next_speaker in range(self.n_actors):
                            # P(Fₜ₊₁|Fₜ)
                            trans_log_prob = 0.0
                            for actor in range(self.n_actors):
                                trans_log_prob += np.log(self.A_F_[actor, f[actor], next_f[actor]])
                            
                            # P(Sₜ₊₁|Sₜ, Fₜ₊₁)
                            logits = self.A_S_[speaker] + self.gamma2_ * np.array(next_f)
                            trans_log_prob += logits[next_speaker] - logsumexp(logits)
                            
                            # P(F̂ₜ₊₁|Fₜ₊₁)
                            for actor in range(self.n_actors):
                                trans_log_prob += np.log(self.B_F_[actor, next_f[actor], F_hat_seq[t+1, actor]])
                            
                            # P(Ŝₜ₊₁|Sₜ₊₁)
                            trans_log_prob += np.log(self.B_S_[next_speaker, S_hat_seq[t+1]])
                            
                            # 累加后向概率
                            log_prob = np.logaddexp(log_prob,
                                                  trans_log_prob + backward_probs[t+1, next_f_idx, next_speaker])
                    
                    backward_probs[t, f_idx, speaker] = log_prob
        
        return backward_probs

    def _compute_expectations(self, X_1, X_2, lengths):
        """计算E步所需的期望值"""
        # 初始化期望计数
        exp_counts = {
            'F_initial': np.zeros(self.n_actors),  # E[N(F·,1,ρ=1)]
            'F_trans': np.zeros((self.n_actors, 2, 2)),  # E[N(F·,·-1,ρ=δ, F·,·,ρ=δ')]
            'FS_initial': np.zeros((self.n_face_states, self.n_actors)),  # E[N(F·,1,·=f, S·,1=ρ)]
            'FSS_trans': np.zeros((self.n_face_states, self.n_actors, self.n_actors)),  # E[N(F·,·,·=f, S·,·-1=ρ, S·,·=ρ')]
            'F_emission': np.zeros((self.n_actors, 2, 2)),  # E[N(F·,·,ρ=δ, F̂·,·,ρ=δ')]
            'S_emission': np.zeros((self.n_actors, self.n_actors)),  # E[N(S·,·=ρ, Ŝ·,·=ρ')]
        }
        
        S_hat = np.argmax(X_1, axis=1)
        face_configs = self._enumerate_face_configs()
        
        start_idx = 0
        for length in lengths:
            end_idx = start_idx + length
            S_hat_seq = S_hat[start_idx:end_idx]
            F_hat_seq = X_2[start_idx:end_idx]
            
            # 计算前向后向概率
            forward_probs = self._forward(F_hat_seq, S_hat_seq, face_configs)
            backward_probs = self._backward(F_hat_seq, S_hat_seq, face_configs)
            
            # 计算序列总概率
            seq_log_prob = logsumexp(forward_probs[-1].flatten())
            
            # 计算各种期望
            self._accumulate_expectations(exp_counts, forward_probs, backward_probs, 
                                        F_hat_seq, S_hat_seq, face_configs, seq_log_prob)
            
            start_idx = end_idx
        
        return exp_counts

    def _accumulate_expectations(self, exp_counts, forward_probs, backward_probs, 
                               F_hat_seq, S_hat_seq, face_configs, seq_log_prob):
        """累积期望计数"""
        T = len(F_hat_seq)
        
        # 计算 gamma(t, f, s) = P(F_t=f, S_t=s | obs)
        for t in range(T):
            for f_idx, f in enumerate(face_configs):
                for speaker in range(self.n_actors):
                    gamma = np.exp(forward_probs[t, f_idx, speaker] + 
                                 backward_probs[t, f_idx, speaker] - seq_log_prob)
                    
                    # 累积面部初始状态期望 (t=0)
                    if t == 0:
                        for actor in range(self.n_actors):
                            if f[actor] == 1:
                                exp_counts['F_initial'][actor] += gamma
                        
                        # 累积联合初始状态期望
                        exp_counts['FS_initial'][f_idx, speaker] += gamma
                    
                    # 累积发射期望
                    for actor in range(self.n_actors):
                        exp_counts['F_emission'][actor, f[actor], F_hat_seq[t, actor]] += gamma
                    
                    exp_counts['S_emission'][speaker, S_hat_seq[t]] += gamma
        
        # 计算转移期望
        for t in range(1, T):
            for f_idx, f in enumerate(face_configs):
                for speaker in range(self.n_actors):
                    for prev_f_idx, prev_f in enumerate(face_configs):
                        for prev_speaker in range(self.n_actors):
                            # 计算 xi(t-1, t)
                            log_xi = (forward_probs[t-1, prev_f_idx, prev_speaker] + 
                                    backward_probs[t, f_idx, speaker] - seq_log_prob)
                            
                            # 添加转移概率
                            for actor in range(self.n_actors):
                                log_xi += np.log(self.A_F_[actor, prev_f[actor], f[actor]])
                            
                            logits = self.A_S_[prev_speaker] + self.gamma2_ * np.array(f)
                            log_xi += logits[speaker] - logsumexp(logits)
                            
                            for actor in range(self.n_actors):
                                log_xi += np.log(self.B_F_[actor, f[actor], F_hat_seq[t, actor]])
                            
                            log_xi += np.log(self.B_S_[speaker, S_hat_seq[t]])
                            
                            xi = np.exp(log_xi)
                            
                            # 累积面部转移期望
                            for actor in range(self.n_actors):
                                exp_counts['F_trans'][actor, prev_f[actor], f[actor]] += xi
                            
                            # 累积说话人转移期望
                            exp_counts['FSS_trans'][f_idx, prev_speaker, speaker] += xi

    def _update_parameters(self, exp_counts, m_sequences):
        """M步：更新参数"""
        if 'a' in self.params:
            # 更新 α
            self.alpha_ = exp_counts['F_initial'] / m_sequences
        
        if 'b' in self.params:
            # 更新 A_F
            for actor in range(self.n_actors):
                for prev_state in range(2):
                    total = exp_counts['F_trans'][actor, prev_state].sum()
                    if total > 0:
                        self.A_F_[actor, prev_state] = exp_counts['F_trans'][actor, prev_state] / total
        
        if 'g' in self.params:
            # 更新 B_F
            for actor in range(self.n_actors):
                for true_state in range(2):
                    total = exp_counts['F_emission'][actor, true_state].sum()
                    if total > 0:
                        self.B_F_[actor, true_state] = exp_counts['F_emission'][actor, true_state] / total
            
            # 更新 B_S
            for true_speaker in range(self.n_actors):
                total = exp_counts['S_emission'][true_speaker].sum()
                if total > 0:
                    self.B_S_[true_speaker] = exp_counts['S_emission'][true_speaker] / total
        
        # 数值优化更新 β, γ₁
        if 'c' in self.params or 'd' in self.params:
            self._optimize_initial_speaker_params(exp_counts)
        
        # 数值优化更新 A_S, γ₂
        if 'e' in self.params or 'f' in self.params:
            self._optimize_transition_speaker_params(exp_counts)

    def _optimize_initial_speaker_params(self, exp_counts):
        """数值优化 β 和 γ₁"""
        face_configs = self._enumerate_face_configs()
        
        def objective(params):
            beta = params[:self.n_actors]
            gamma1 = params[self.n_actors]
            
            log_likelihood = 0.0
            for f_idx, f in enumerate(face_configs):
                for speaker in range(self.n_actors):
                    count = exp_counts['FS_initial'][f_idx, speaker]
                    if count > 0:
                        logits = beta + gamma1 * np.array(f)
                        log_prob = logits[speaker] - logsumexp(logits)
                        log_likelihood += count * log_prob
            
            return -log_likelihood
        
        # 初始值
        x0 = np.concatenate([self.beta_, [self.gamma1_]])
        
        # 优化
        result = minimize(objective, x0, method='BFGS')
        
        if result.success:
            self.beta_ = result.x[:self.n_actors]
            self.gamma1_ = result.x[self.n_actors]

    def _optimize_transition_speaker_params(self, exp_counts):
        """数值优化 A_S 和 γ₂"""
        face_configs = self._enumerate_face_configs()
        
        def objective(params):
            A_S = params[:-1].reshape(self.n_actors, self.n_actors)
            gamma2 = params[-1]
            
            log_likelihood = 0.0
            for f_idx, f in enumerate(face_configs):
                for prev_speaker in range(self.n_actors):
                    for speaker in range(self.n_actors):
                        count = exp_counts['FSS_trans'][f_idx, prev_speaker, speaker]
                        if count > 0:
                            logits = A_S[prev_speaker] + gamma2 * np.array(f)
                            log_prob = logits[speaker] - logsumexp(logits)
                            log_likelihood += count * log_prob
            
            return -log_likelihood
        
        # 初始值
        x0 = np.concatenate([self.A_S_.flatten(), [self.gamma2_]])
        
        # 优化
        result = minimize(objective, x0, method='BFGS')
        
        if result.success:
            self.A_S_ = result.x[:-1].reshape(self.n_actors, self.n_actors)
            self.gamma2_ = result.x[-1]

    def fit(self, X_1, X_2, lengths=None):
        """训练嵌套HMM模型"""
        X_1 = np.array(X_1)
        X_2 = np.array(X_2)
        
        self._check_and_set_n_features(X_1, X_2)
        lengths = self._validate_lengths(X_1, lengths)
        
        # 初始化参数
        self._init_params()
        
        # EM迭代
        self.monitor_._reset()
        
        for n_iter in range(self.n_iter):
            # E步：计算期望
            exp_counts = self._compute_expectations(X_1, X_2, lengths)
            
            # M步：更新参数
            self._update_parameters(exp_counts, len(lengths))
            
            # 计算对数似然
            curr_log_prob = self._compute_log_likelihood(X_1, X_2, lengths)
            
            # 检查收敛
            self.monitor_.history.append(curr_log_prob)
            self.monitor_.iter = n_iter
            self.monitor_.report(curr_log_prob)
            
            if self.monitor_.converged:
                break
        
        return self

    def score(self, X_1, X_2, lengths=None):
        """计算给定观测序列的对数似然"""
        X_1 = np.array(X_1)
        X_2 = np.array(X_2)
        
        self._check_and_set_n_features(X_1, X_2)
        lengths = self._validate_lengths(X_1, lengths)
        
        return self._compute_log_likelihood(X_1, X_2, lengths)

    def predict(self, X_1, X_2, lengths=None):
        """预测最可能的隐状态序列"""
        # 实现Viterbi算法
        # ... (可选实现)
        pass

    @property
    def converged(self):
        """检查模型是否收敛"""
        return self.monitor_.converged