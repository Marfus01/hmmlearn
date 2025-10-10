import os, sys
import numpy as np
from hmmlearn import hmm

# used to ignore the warning of hmmlearn
class SuppressMultinomialHMMWarning:
    def __enter__(self):
        self._original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self._original_stderr



# 设置随机种子以便复现
np.random.seed(42)

# 定义状态数量
n_states_hid = 6  # 隐藏状态数
n_states_obs = 6  # 观察状态数
n_samples = 30    # 样本数量

# 生成真实的HMM参数
true_startprob = np.random.dirichlet(np.ones(n_states_hid)) # 初始状态分布
true_transmat = np.random.dirichlet(np.ones(n_states_hid), size=n_states_hid) # 状态转移概率矩阵
true_emissionprob = np.random.dirichlet(np.ones(n_states_obs), size=n_states_hid) # 发射概率矩阵
for i in range(n_states_hid):
    true_transmat[i, i] += 1  # 增加主对角线的权重
    true_transmat[i] /= true_transmat[i].sum()  # 归一化

for i in range(n_states_hid):
    true_emissionprob[i, i % n_states_obs] += 1  # 增加主对角线的权重
    true_emissionprob[i] /= true_emissionprob[i].sum()  # 归一化
print("真实的初始状态分布:", true_startprob)
print("真实的状态转移概率矩阵:\n", true_transmat)
print("真实的发射概率矩阵:\n", true_emissionprob)

# 生成样本数据
## 函数
def generate_hmm_samples(startprob, transmat, emissionprob, n_samples):
    """生成HMM样本数据"""
    hidden_states = []
    observations = []
    
    # 生成初始状态
    current_state = np.random.choice(n_states_hid, p=startprob)
    
    for _ in range(n_samples):
        # 记录当前隐藏状态
        hidden_states.append(current_state)
        
        # 根据发射概率生成观察
        obs = np.random.choice(n_states_obs, p=emissionprob[current_state])
        observations.append(obs)
        
        # 转移到下一个状态
        current_state = np.random.choice(n_states_hid, p=transmat[current_state])
    
    return np.array(hidden_states), np.array(observations)

## 生成训练数据
true_hidden_states, speaker_obs_raw = generate_hmm_samples(
    true_startprob, true_transmat, true_emissionprob, n_samples
)

## 设置样本长度（所有观察序列作为一个整体）
lengths = [20, 10]

## 将观察数据转换为MultinomialHMM期望的格式（one-hot编码）
speaker_obs = np.zeros((n_samples, n_states_obs), dtype=int)
for i, obs in enumerate(speaker_obs_raw):
    speaker_obs[i, obs] = 1




# define initial probs in hmm
audio_startprob_init = np.ones(n_states_hid) / n_states_hid  # uniform distribution, of shape (n_states,)
audio_transitionprob_init = np.ones((n_states_hid, n_states_hid)) * (1 - 0.4) / n_states_hid
audio_emissionprob_init = np.ones((n_states_hid, n_states_obs)) * 0.4 / n_states_obs
# round these probs to 5 decimal places
audio_startprob_init = np.round(audio_startprob_init, 5)
audio_transitionprob_init = np.round(audio_transitionprob_init, 5)
audio_emissionprob_init = np.round(audio_emissionprob_init, 5)
# make sure the sum of each row is 1
audio_startprob_init[0] = 1 - np.sum(audio_startprob_init[1:])
for i in range(n_states_hid):
    audio_transitionprob_init[i, i] = 1 - np.sum(audio_transitionprob_init[i]) + audio_transitionprob_init[i, i]
    audio_emissionprob_init[i, i] = 1 - np.sum(audio_emissionprob_init[i]) + audio_emissionprob_init[i, i]


# 对比 MultinomialHMM(n_trials=1) 和 CategoricalHMM
print("\n=== 模型对比实验 ===")

# 1. MultinomialHMM with n_trials=1
with SuppressMultinomialHMMWarning():
    multinomial_model = hmm.MultinomialHMM(n_components=n_states_hid, 
                                n_trials=1,
                                n_iter=1000, tol=0.00001,
                                init_params='')

multinomial_model.n_features = n_states_obs
multinomial_model.startprob_ = audio_startprob_init.copy()
multinomial_model.transmat_ = audio_transitionprob_init.copy()
multinomial_model.emissionprob_ = audio_emissionprob_init.copy()

multinomial_model.fit(speaker_obs, lengths)
pred_multinomial = multinomial_model.predict(speaker_obs, lengths)
pred_multinomial_prob = multinomial_model.predict_proba(speaker_obs, lengths)

# 2. CategoricalHMM
with SuppressMultinomialHMMWarning():
    categorical_model = hmm.CategoricalHMM(n_components=n_states_hid,
                                        n_iter=1000, tol=0.00001,
                                        init_params='')

categorical_model.startprob_ = audio_startprob_init.copy()
categorical_model.transmat_ = audio_transitionprob_init.copy()
categorical_model.emissionprob_ = audio_emissionprob_init.copy()

# 注意：CategoricalHMM需要的数据格式是 (n_samples, 1)
speaker_obs_categorical = speaker_obs_raw.reshape(-1, 1)
categorical_model.fit(speaker_obs_categorical, lengths)
pred_categorical = categorical_model.predict(speaker_obs_categorical, lengths)
pred_categorical_prob = categorical_model.predict_proba(speaker_obs_categorical, lengths)

# 比较结果
print("\n=== MultinomialHMM (n_trials=1) 结果 ===")
print("观察序列:", speaker_obs_raw)
print("真实的隐藏状态:", true_hidden_states)
print("预测的隐藏状态:", pred_multinomial)
print("学习后的转移概率矩阵:")
print(multinomial_model.transmat_)
print("学习后的发射概率矩阵:")
print(multinomial_model.emissionprob_)

# # 计算准确率（考虑到状态标签可能不对应，这里只是简单比较）
# print(f"\n后验概率矩阵形状: {pred_speaker_hmm_prob.shape}")
# print("每个时刻的最大后验概率:")
# max_probs = np.max(pred_speaker_hmm_prob, axis=1)
# for i, prob in enumerate(max_probs):
#     print(f"时刻 {i}: 状态 {pred_speaker_hmm[i]}, 概率 {prob:.4f}")

print("\n=== CategoricalHMM 结果 ===")
print("观察序列:", speaker_obs_raw)
print("预测的隐藏状态:", pred_categorical)
print("学习后的转移概率矩阵:")
print(categorical_model.transmat_)
print("学习后的发射概率矩阵:")
print(categorical_model.emissionprob_)

# 对比分析
print("\n=== 对比分析 ===")
print("隐状态预测是否一致:", np.array_equal(pred_multinomial, pred_categorical))
print("转移概率矩阵是否一致:", np.allclose(multinomial_model.transmat_, categorical_model.transmat_, rtol=1e-5))
print("发射概率矩阵是否一致:", np.allclose(multinomial_model.emissionprob_, categorical_model.emissionprob_, rtol=1e-5))
print("后验概率是否一致:", np.allclose(pred_multinomial_prob, pred_categorical_prob, rtol=1e-5))

if not np.array_equal(pred_multinomial, pred_categorical):
    print("\n隐状态预测差异:")
    for i, (m, c) in enumerate(zip(pred_multinomial, pred_categorical)):
        if m != c:
            print(f"时刻 {i}: Multinomial={m}, Categorical={c}")

print(f"\n后验概率差异 (最大绝对差值): {np.max(np.abs(pred_multinomial_prob - pred_categorical_prob)):.6f}")