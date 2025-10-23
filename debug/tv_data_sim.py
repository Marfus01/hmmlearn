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
    - X_onehot: one-hot 协变量矩阵 (n_frames, n_actors)，对应于每一帧的活跃说话人
    """
    n_actors = len(actors)
    
    # 初始化序列
    F = np.zeros((n_frames, n_actors), dtype=int)
    S = np.zeros(n_frames, dtype=int)
    F_hat = np.zeros((n_frames, n_actors), dtype=int)
    S_hat = np.zeros(n_frames, dtype=int)
    X_onehot = np.zeros((n_frames, n_actors), dtype=int)
    
    # 1. 生成面部出现序列 F (对每个演员独立的二元马尔可夫链)
    for actor_id in actors:
        # 初始状态
        F[0, actor_id] = np.random.binomial(1, params['alpha'][actor_id])
        
        # 后续状态转移
        for t in range(1, n_frames):
            prev_state = F[t-1, actor_id]
            transition_prob = params['A_F'][actor_id][prev_state, 1]  # 转移到状态1的概率
            F[t, actor_id] = np.random.binomial(1, transition_prob)
    
    # 1.5 生成协变量 X_onehot (独立生成，可以认为是某种外部先验信息)
    x_probs = np.array([0.4, 0.3, 0.3])    # 协变量生成概率 (任意时刻每个演员为活跃说话人的先验概率)
    for t in range(n_frames):
        # 每个时刻，以一定概率生成活跃说话人
        # 这里假设协变量独立于F，在实际应用中可能有其他生成机制
        x_actor = np.random.choice(actors, size=1, p=x_probs)[0]
        X_onehot[t] = np.eye(n_actors)[x_actor]
    
    # 2. 生成说话人序列 S (依赖于面部出现和协变量X)
    for t in range(n_frames):
        if t == 0:
            # 初始状态概率，依赖于面部出现和协变量X
            logits = np.array([params['beta'][actor_id] + 
                              params['gamma1'] * F[t, actor_id] + 
                              params['eta1'] * X_onehot[t, actor_id]
                              for actor_id in actors])
            probs = softmax(logits)
            S[t] = np.random.choice(actors, p=probs)
        else:
            # 转移概率，依赖于前一状态、当前面部出现和协变量X
            prev_speaker = S[t-1]
            logits = np.array([params['A_S'][prev_speaker, actor_id] + 
                              params['gamma2'] * F[t, actor_id] + 
                              params['eta2'] * X_onehot[t, actor_id]
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
    
    return F, S, F_hat, S_hat, X_onehot

def generate_multiple_sequences(params, sequence_lengths, actors):
    """
    生成多个不同长度的序列，返回新的数据格式
    
    返回:
    - S_hat_onehot: 说话人标签 (n_total_samples, n_actors) one-hot编码
    - F_hat: 面部出现 (n_total_samples, n_actors) 
    - X_onehot: one-hot 协变量 (n_total_samples, n_actors)
    - lengths: 序列长度列表
    """
    sequences = []
    S_hat_onehot_list = []
    F_hat_list = []
    X_list = []
    F_true_list = []
    S_true_list = []
    
    for seq_id, length in enumerate(sequence_lengths):
        F, S, F_hat, S_hat, X_onehot = generate_nested_markov_sequence(params, length, actors)
        
        # 将S_hat转换为one-hot编码
        S_hat_onehot = np.zeros((length, len(actors)))
        S_hat_onehot[np.arange(length), S_hat] = 1
        
        S_hat_onehot_list.append(S_hat_onehot)  # 说话人标签 (one-hot)
        F_hat_list.append(F_hat)         # 面部出现
        X_list.append(X_onehot)                 # 协变量
        F_true_list.append(F)          # 真实面部状态
        S_true_list.append(S)          # 真实说话人状态
        
        sequences.append({
            'sequence_id': seq_id,
            'length': length,
            'F': F, 'S': S, 'F_hat': F_hat, 'S_hat': S_hat, 'X_onehot': X_onehot,
            'S_hat_onehot': S_hat_onehot, 'F_hat': F_hat
        })
    
    # 合并所有序列
    S_hat_onehot = np.vstack(S_hat_onehot_list)
    F_hat = np.vstack(F_hat_list)
    X_onehot = np.vstack(X_list)
    F_true = np.vstack(F_true_list)
    S_true = np.hstack(S_true_list)
    
    true_states = {
        'face_states': F_true,
        'speaker_states': S_true
    }
    
    return S_hat_onehot, F_hat, X_onehot, sequence_lengths, sequences, true_states

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
    'eta1': 2,    # 协变量对初始说话人的影响
    'eta2': 2,    # 协变量对说话人转移的影响
    
    # 说话人转移偏好矩阵 (仅依赖音频的部分)
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

params['beta'] -= params['beta'][0]  # 归一化基线
params['A_S'] -= np.diag(params['A_S'])[:,None]    # 固定转移到自己的logit为0，作为基准

# 生成多个不同长度的序列
sequence_lengths = [150, 200, 120, 250, 180]
S_hat_onehot, F_hat, X_onehot, lengths, sequences, true_states = generate_multiple_sequences(params, sequence_lengths, actors)

# # 打印结果
# print("=== 嵌套马尔可夫模型数据生成结果 ===")
# print(f"生成了 {len(sequences)} 个序列")
# print(f"演员数量: {n_actors}")

# print(f"\n=== 模型参数 ===")
# print(f"面部出现初始概率 α: {params['alpha']}")
# print(f"面部影响参数 γ1: {params['gamma1']}, γ2: {params['gamma2']}")
print(f"活跃说话人影响参数 η1 (初始): {params['eta1']}, η2 (转移): {params['eta2']}")

print("=== 数据格式验证 ===")
print(f"S_hat_onehot shape: {S_hat_onehot.shape} (说话人标签, one-hot)")
print(f"F_hat shape: {F_hat.shape} (面部出现)")
print(f"X_onehot shape: {X_onehot.shape} (协变量)")
print(f"lengths: {lengths}")

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

print("\n=== 协变量X与真实说话人的关系分析 ===")
for actor in actors:
    # 计算X=1/0时，是说话人的比例
    x_is_one = (X_onehot[:, actor] == 1)
    x_is_zero = (X_onehot[:, actor] == 0)
    is_speaker = (true_states['speaker_states'] == actor)

    ratio_speaker_when_x1 = np.mean(is_speaker[x_is_one]) if x_is_one.sum() > 0 else 0
    ratio_speaker_when_x0 = np.mean(is_speaker[x_is_zero]) if x_is_zero.sum() > 0 else 0

    print(f"演员 {actor}:")
    print(f"  X=1 时是说话人的比例: {ratio_speaker_when_x1:.3f}")
    print(f"  X=0 时是说话人的比例: {ratio_speaker_when_x0:.3f}")

# 创建和训练模型
from hmmlearn.nested_hmm import NestedHMM
from hmmlearn.nested_hmm_full import NestedHMM_full
import time

def run_hmm_analysis(S_hat_onehot, F_hat, X_onehot, lengths, model_name="NestedHMM", 
                     true_states=true_states, params=params, n_actors=3, n_iter=10, tol=1e-3, verbose=True):
    """
    通用HMM分析流程：模型选择、拟合、后验概率、解码、准确率统计
    参数:
        S_hat_onehot, F_hat, X_onehot, lengths: 数据
        model_name: "NestedHMM"等
        true_states, params: 真实状态和参数（用于对比）
        n_actors, n_iter, tol, verbose: 模型参数
    """
    print(f"\n=== 使用模型: {model_name} ===")
    print("\n=== 训练模型 ===")
    start_time = time.time()
    print("训练开始时间:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))

    if model_name == "NestedHMM":
        model = NestedHMM(n_actors=n_actors, n_iter=n_iter, tol=tol, verbose=verbose)
        model.fit(S_hat_onehot, F_hat, lengths)
    elif model_name == "NestedHMM_Full":
        model = NestedHMM_full(n_actors=n_actors, n_iter=n_iter, tol=tol, verbose=verbose)
        model.fit(S_hat_onehot, F_hat, X_onehot, lengths)

    end_time = time.time()
    print("训练结束时间:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
    print("训练耗时:", end_time - start_time, "秒")

    # # 对比模型参数的真实值和拟合值
    # print("\n=== 学习到的参数对比 ===")
    # print("α (面部出现初始概率):")
    # print("真实:", np.round(params['alpha'], 3))
    # print("学习:", np.round(model.alpha_, 3))

    # print("\nβ (说话人初始偏好):")
    # print("真实:", np.round(params['beta'], 3))
    # print("学习:", np.round(model.beta_, 3))

    # print("\nγ1 (面部对说话人初始影响):")
    # print("真实:", np.round(params['gamma1'], 3))
    # print("学习:", np.round(model.gamma1_, 3))

    # print("\nγ2 (面部对说话人转移影响):")
    # print("真实:", np.round(params['gamma2'], 3))
    # print("学习:", np.round(model.gamma2_, 3))

    # print("\nA_F (面部转移矩阵):")
    # for i in range(n_actors):
    #     print(f"演员{i} 真实:", np.round(params['A_F'][i], 3))
    #     print(f"演员{i} 学习:", np.round(model.A_F_[i], 3))

    # print("\nA_S (说话人转移矩阵):")
    # print("真实:", np.round(params['A_S'], 3))
    # print("学习:", np.round(model.A_S_, 3))

    # print("\nB_F (面部识别混淆矩阵):")
    # for i in range(n_actors):
    #     print(f"演员{i} 真实:", np.round(params['B_F'][i], 3))
    #     print(f"演员{i} 学习:", np.round(model.B_F_[i], 3))

    # print("\nB_S (说话人识别混淆矩阵):")
    # print("真实:", np.round(params['B_S'], 3))
    # print("学习:", np.round(model.B_S_, 3))

    # 计算后验概率
    print("\n=== 计算后验概率 ===")
    if model_name == "NestedHMM":
        pred_probs = model.predict_proba(S_hat_onehot, F_hat, lengths)
    elif model_name == "NestedHMM_Full":
        pred_probs = model.predict_proba(S_hat_onehot, F_hat, X_onehot, lengths)
    print(pred_probs['joint_states'][-1])

    # viterbi 解码结果
    print("\n=== Viterbi 解码结果 ===")
    start_time = time.time()
    if model_name == "NestedHMM":
        face_states_viterbi, speaker_states_viterbi = model.predict(S_hat_onehot, F_hat, lengths)
    elif model_name == "NestedHMM_Full":
        face_states_viterbi, speaker_states_viterbi = model.predict(S_hat_onehot, F_hat, X_onehot, lengths)
    end_time = time.time()
    print("viterbi解码耗时:", end_time - start_time, "秒")

    # 计算准确率
    print("\n=== 准确率统计 ===")
    ## 面部状态准确率
    face_acc_viterbi = np.mean(true_states['face_states'] == face_states_viterbi)
    face_acc_observed = np.mean(true_states['face_states'] == F_hat)
    ## 说话人状态准确率
    speaker_acc_viterbi = np.mean(true_states['speaker_states'] == speaker_states_viterbi)
    speaker_acc_observed = np.mean(true_states['speaker_states'] == np.argmax(S_hat_onehot, axis=1))

    print("\n=== 准确率分析 ===")
    print(f"面部状态准确率:")
    print(f"  观测准确率: {face_acc_observed:.3f}")
    print(f"  Viterbi准确率: {face_acc_viterbi:.3f}")

    print(f"说话人状态准确率:")
    print(f"  观测准确率: {speaker_acc_observed:.3f}")
    print(f"  Viterbi准确率: {speaker_acc_viterbi:.3f}")

# 运行分析
run_hmm_analysis(S_hat_onehot, F_hat, X_onehot=None, lengths=lengths, model_name="NestedHMM", 
                 true_states=true_states, params=params, n_actors=n_actors, n_iter=50, tol=1e-3, verbose=True)
run_hmm_analysis(S_hat_onehot, F_hat, X_onehot, lengths=lengths, model_name="NestedHMM_Full", 
                 true_states=true_states, params=params, n_actors=n_actors, n_iter=50, tol=1e-3, verbose=True)