import numpy as np
from scipy.special import softmax

def generate_nested_markov_sequence(params, n_frames, actors):
    """
    生成一个视频序列的嵌套马尔可夫模型数据
    
    参数:
    - params: 模型参数字典
    - n_frames: 序列长度
    - actors: 演员列表 (例如 [0, 1, 2] 表示3个演员，必须为从0开始的连续整数)
    
    返回:
    - F: 面部出现矩阵 (n_frames, n_actors)
    - S: 说话人标签 (n_frames,)
    - F_hat: 预测的面部出现矩阵 (n_frames, n_actors)
    - S_hat: 预测的说话人标签 (n_frames,)
    """
    n_actors = len(actors)
    
    # 初始化序列
    F = np.zeros((n_frames, n_actors), dtype=int)
    S = np.zeros(n_frames, dtype=int)
    F_hat = np.zeros((n_frames, n_actors), dtype=int)
    S_hat = np.zeros(n_frames, dtype=int)
    
    # 1. 生成面部出现序列 F (对每个演员独立的二元马尔可夫链)
    for actor_id in actors:
        # 初始状态
        F[0, actor_id] = np.random.binomial(1, params['alpha'][actor_id])
        
        # 后续状态转移
        for t in range(1, n_frames):
            prev_state = F[t-1, actor_id]
            transition_prob = params['A_F'][actor_id][prev_state, 1]  # 转移到状态1的概率
            F[t, actor_id] = np.random.binomial(1, transition_prob)
    
    # 2. 生成说话人序列 S (依赖于面部出现)
    for t in range(n_frames):
        if t == 0:
            # 初始状态概率，依赖于面部出现
            logits = np.array([params['beta'][actor_id] + params['gamma1'] * F[t, actor_id] 
                              for actor_id in actors])
            probs = softmax(logits)
            S[t] = np.random.choice(actors, p=probs)
        else:
            # 转移概率，依赖于前一状态和当前面部出现
            prev_speaker = S[t-1]
            logits = np.array([params['A_S'][prev_speaker, actor_id] + params['gamma2'] * F[t, actor_id] 
                              for actor_id in actors])
            probs = softmax(logits)
            S[t] = np.random.choice(actors, p=probs)
    
    # 3. 生成观测值 F_hat (通过混淆矩阵)
    for t in range(n_frames):
        for actor_id in actors:
            true_state = F[t, actor_id]
            confusion_prob = params['B_F'][actor_id][true_state, 1]  # 预测为1的概率
            F_hat[t, actor_id] = np.random.binomial(1, confusion_prob)
    
    # 4. 生成观测值 S_hat (通过混淆矩阵)
    for t in range(n_frames):
        true_speaker = S[t]
        probs = params['B_S'][true_speaker, :]
        S_hat[t] = np.random.choice(actors, p=probs)
    
    return F, S, F_hat, S_hat

def generate_multiple_sequences(params, sequence_lengths, actors):
    """
    生成多个不同长度的序列，返回新的数据格式
    
    返回:
    - X_1: 说话人标签 (n_total_samples, n_actors) one-hot编码
    - X_2: 面部出现 (n_total_samples, n_actors) 
    - lengths: 序列长度列表
    """
    sequences = []
    X_1_list = []
    X_2_list = []
    
    for seq_id, length in enumerate(sequence_lengths):
        F, S, F_hat, S_hat = generate_nested_markov_sequence(params, length, actors)
        
        # 将S_hat转换为one-hot编码
        S_hat_onehot = np.zeros((length, len(actors)))
        S_hat_onehot[np.arange(length), S_hat] = 1
        
        X_1_list.append(S_hat_onehot)  # 说话人标签 (one-hot)
        X_2_list.append(F_hat)         # 面部出现
        
        sequences.append({
            'sequence_id': seq_id,
            'length': length,
            'F': F, 'S': S, 'F_hat': F_hat, 'S_hat': S_hat,
            'X_1': S_hat_onehot, 'X_2': F_hat
        })
    
    # 合并所有序列
    X_1 = np.vstack(X_1_list)
    X_2 = np.vstack(X_2_list)
    
    return X_1, X_2, sequence_lengths, sequences

# 设置随机种子
np.random.seed(42)

# 定义演员和参数
actors = [0, 1, 2]  # 3个演员
n_actors = len(actors)

# 定义模型参数
params = {
    # 面部出现的初始概率 (每个演员)
    'alpha': np.array([0.3, 0.5, 0.2]),
    
    # 面部出现的转移矩阵 (每个演员的2x2矩阵)
    'A_F': [
        np.array([[0.8, 0.2], [0.3, 0.7]]),  # 演员0
        np.array([[0.7, 0.3], [0.25, 0.75]]), # 演员1
        np.array([[0.85, 0.15], [0.4, 0.6]])  # 演员2
    ],
    
    # 说话人的基础参数
    'beta': np.array([0.1, 0.2, 0.15]),  # 基础偏好
    'gamma1': 2.0,  # 面部出现对初始说话人的影响
    'gamma2': 1.5,  # 面部出现对说话人转移的影响
    
    # 说话人转移矩阵 (仅依赖音频的部分)
    'A_S': np.array([
        [0.6, 0.25, 0.15],
        [0.1, 0.7, 0.2],
        [0.3, 0.2, 0.5]
    ]),
    
    # 面部识别混淆矩阵 (每个演员)
    'B_F': [
        np.array([[0.9, 0.1], [0.15, 0.85]]),  # 演员0
        np.array([[0.85, 0.15], [0.2, 0.8]]),   # 演员1
        np.array([[0.88, 0.12], [0.18, 0.82]])  # 演员2
    ],
    
    # 说话人识别混淆矩阵
    'B_S': np.array([
        [0.8, 0.15, 0.05],
        [0.1, 0.85, 0.05],
        [0.1, 0.1, 0.8]
    ])
}

# 生成多个不同长度的序列
sequence_lengths = [15, 20, 12, 25, 18]
X_1, X_2, lengths, sequences = generate_multiple_sequences(params, sequence_lengths, actors)

# # 打印结果
# print("=== 嵌套马尔可夫模型数据生成结果 ===")
# print(f"生成了 {len(sequences)} 个序列")
# print(f"演员数量: {n_actors}")

# for i, seq in enumerate(sequences):
#     print(f"\n--- 序列 {i+1} (长度: {seq['length']}) ---")
#     print(f"真实面部出现 F:\n{seq['F']}")
#     print(f"真实说话人 S: {seq['S']}")
#     print(f"预测面部出现 F_hat:\n{seq['F_hat']}")
#     print(f"预测说话人 S_hat: {seq['S_hat']}")
    
#     # 计算一些统计信息
#     face_accuracy = np.mean(seq['F'] == seq['F_hat'])
#     speaker_accuracy = np.mean(seq['S'] == seq['S_hat'])
#     print(f"面部识别准确率: {face_accuracy:.3f}")
#     print(f"说话人识别准确率: {speaker_accuracy:.3f}")

# print(f"\n=== 模型参数 ===")
# print(f"面部出现初始概率 α: {params['alpha']}")
# print(f"面部影响参数 γ1: {params['gamma1']}, γ2: {params['gamma2']}")

print("=== 数据格式验证 ===")
print(f"X_1 shape: {X_1.shape} (说话人标签, one-hot)")
print(f"X_2 shape: {X_2.shape} (面部出现)")
print(f"lengths: {lengths}")

# 创建和训练模型
from hmmlearn.nested_hmm import NestedHMM
print("\n=== 训练模型 ===")
model = NestedHMM(n_actors=3, n_iter=20, verbose=True, tol=1e-3)
model.fit(X_1, X_2, lengths)

# 计算似然
log_likelihood = model.score(X_1, X_2, lengths)
print(f"\nFinal log likelihood: {log_likelihood}")

# 输出学习到的参数
print("\n=== 学习到的参数 ===")
print(f"真实 α: {params['alpha']}")
print(f"学习 α: {model.alpha_}")
print(f"真实 γ1: {params['gamma1']}")
print(f"学习 γ1: {model.gamma1_}")
print(f"真实 γ2: {params['gamma2']}")
print(f"学习 γ2: {model.gamma2_}")