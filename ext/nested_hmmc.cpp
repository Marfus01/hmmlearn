#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

namespace py = pybind11;
using ssize_t = Py_ssize_t;

// 数值稳定的 logaddexp 函数
double logaddexp(double a, double b) {
    if (a == -std::numeric_limits<double>::infinity()) return b;
    if (b == -std::numeric_limits<double>::infinity()) return a;
    return std::max(a, b) + std::log1p(std::exp(-std::abs(b - a)));
}

// 数值稳定的 logsumexp 函数
double logsumexp(const std::vector<double>& values) {
    if (values.empty()) return -std::numeric_limits<double>::infinity();
    
    double max_val = *std::max_element(values.begin(), values.end());
    if (std::isinf(max_val)) return max_val;
    
    double sum = 0.0;
    for (double val : values) {
        sum += std::exp(val - max_val);
    }
    return std::log(sum) + max_val;
}

// 添加logsumexp的重载版本
double logsumexp(const double* values, size_t size) {
    if (size == 0) return -std::numeric_limits<double>::infinity();
    
    double max_val = *std::max_element(values, values + size);
    if (std::isinf(max_val)) return max_val;
    
    double sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        sum += std::exp(values[i] - max_val);
    }
    return std::log(sum) + max_val;
}

// 嵌套HMM前向算法
py::array_t<double> nested_forward_pass(
    py::array_t<int> X_1,                // 说话人观测 [n_samples, n_actors]
    py::array_t<int> X_2,                // 面部出现 [n_samples, n_actors]  
    py::array_t<double> alpha,           // 初始状态概率 [n_actors]
    py::array_t<double> A_F,             // 面部转移矩阵 [n_actors, 2, 2]
    py::array_t<double> beta,            // 说话人初始概率 [n_actors]
    py::array_t<double> A_S,             // 说话人转移矩阵 [n_actors, n_actors]
    py::array_t<double> B_F,             // 面部观测矩阵 [n_actors, 2, 2]
    py::array_t<double> B_S,             // 说话人观测矩阵 [n_actors, n_actors]
    double gamma1,
    double gamma2
) {
    auto X_1_ptr = X_1.template unchecked<2>();
    auto X_2_ptr = X_2.template unchecked<2>();  
    auto alpha_ptr = alpha.template unchecked<1>();
    auto A_F_ptr = A_F.template unchecked<3>();
    auto beta_ptr = beta.template unchecked<1>();
    auto A_S_ptr = A_S.template unchecked<2>();
    auto B_F_ptr = B_F.template unchecked<3>();
    auto B_S_ptr = B_S.template unchecked<2>();
    
    int T = X_1.shape(0);
    int n_actors = X_1.shape(1);
    int n_face_states = 1 << n_actors;  // 2^n_actors
    
    // 前向变量 U[t, f, rho]
    auto U = py::array_t<double>({T, n_face_states, n_actors});
    auto U_ptr = U.template mutable_unchecked<3>();
    
    {
        py::gil_scoped_release nogil;
        
        // 初始化所有值为负无穷
        for (int t = 0; t < T; ++t) {
            for (int f = 0; f < n_face_states; ++f) {
                for (int rho = 0; rho < n_actors; ++rho) {
                    U_ptr(t, f, rho) = -std::numeric_limits<double>::infinity();
                }
            }
        }
        
        // t=0 初始化
        for (int f = 0; f < n_face_states; ++f) {
            for (int rho = 0; rho < n_actors; ++rho) {
                double log_prob = 0.0;
                
                // P(F_{1,·} = f) = ∏_ρ P(F_{1,ρ})
                for (int actor = 0; actor < n_actors; ++actor) {
                    int face_state = (f >> actor) & 1;
                    double alpha_val = alpha_ptr(actor);
                    if (alpha_val <= 0.0 || alpha_val >= 1.0) {
                        log_prob = -std::numeric_limits<double>::infinity();
                        break;
                    }
                    log_prob += std::log(face_state ? alpha_val : (1.0 - alpha_val));
                }
                
                if (std::isinf(log_prob)) continue;
                
                // P(S_1 = ρ | F_{1,·} = f)
                double beta_val = beta_ptr(rho);
                if (beta_val <= 0.0) continue;
                
                int face_rho = (f >> rho) & 1;
                double speaker_log_prob = std::log(beta_val) + gamma1 * face_rho;
                
                // 归一化说话人概率
                std::vector<double> speaker_logits(n_actors);
                for (int s = 0; s < n_actors; ++s) {
                    int face_s = (f >> s) & 1;
                    speaker_logits[s] = std::log(beta_ptr(s)) + gamma1 * face_s;
                }
                double log_normalizer = logsumexp(speaker_logits);
                speaker_log_prob -= log_normalizer;
                
                // 观测概率
                double obs_log_prob = 0.0;
                
                // 面部观测
                for (int actor = 0; actor < n_actors; ++actor) {
                    int true_face = (f >> actor) & 1;
                    int obs_face = X_2_ptr(0, actor);
                    double b_f = B_F_ptr(actor, true_face, obs_face);
                    if (b_f <= 0.0) {
                        obs_log_prob = -std::numeric_limits<double>::infinity();
                        break;
                    }
                    obs_log_prob += std::log(b_f);
                }
                
                if (std::isinf(obs_log_prob)) continue;
                
                // 说话人观测
                int obs_speaker = -1;
                for (int s = 0; s < n_actors; ++s) {
                    if (X_1_ptr(0, s) == 1) {
                        obs_speaker = s;
                        break;
                    }
                }
                
                if (obs_speaker >= 0) {
                    double b_s = B_S_ptr(rho, obs_speaker);
                    if (b_s > 0.0) {
                        obs_log_prob += std::log(b_s);
                    } else {
                        obs_log_prob = -std::numeric_limits<double>::infinity();
                    }
                }
                
                if (!std::isinf(obs_log_prob)) {
                    U_ptr(0, f, rho) = log_prob + speaker_log_prob + obs_log_prob;
                }
            }
        }
        
        // t > 0 递推
        for (int t = 1; t < T; ++t) {
            for (int f = 0; f < n_face_states; ++f) {
                for (int rho = 0; rho < n_actors; ++rho) {
                    std::vector<double> transition_probs;
                    
                    for (int prev_f = 0; prev_f < n_face_states; ++prev_f) {
                        for (int prev_rho = 0; prev_rho < n_actors; ++prev_rho) {
                            double prev_prob = U_ptr(t-1, prev_f, prev_rho);
                            if (std::isinf(prev_prob)) continue;
                            
                            // 面部转移概率
                            double face_trans_log_prob = 0.0;
                            for (int actor = 0; actor < n_actors; ++actor) {
                                int prev_face = (prev_f >> actor) & 1;
                                int curr_face = (f >> actor) & 1;
                                double a_f = A_F_ptr(actor, prev_face, curr_face);
                                if (a_f <= 0.0) {
                                    face_trans_log_prob = -std::numeric_limits<double>::infinity();
                                    break;
                                }
                                face_trans_log_prob += std::log(a_f);
                            }
                            
                            if (std::isinf(face_trans_log_prob)) continue;
                            
                            // 说话人转移概率
                            double speaker_trans_log_prob = std::log(A_S_ptr(prev_rho, rho)) + 
                                                          gamma2 * ((f >> rho) & 1);
                            
                            // 归一化
                            std::vector<double> speaker_trans_logits(n_actors);
                            for (int s = 0; s < n_actors; ++s) {
                                speaker_trans_logits[s] = std::log(A_S_ptr(prev_rho, s)) + 
                                                        gamma2 * ((f >> s) & 1);
                            }
                            double trans_normalizer = logsumexp(speaker_trans_logits);
                            speaker_trans_log_prob -= trans_normalizer;
                            
                            transition_probs.push_back(prev_prob + face_trans_log_prob + speaker_trans_log_prob);
                        }
                    }
                    
                    if (!transition_probs.empty()) {
                        double max_trans_prob = logsumexp(transition_probs);
                        
                        // 观测概率
                        double obs_log_prob = 0.0;
                        
                        // 面部观测
                        for (int actor = 0; actor < n_actors; ++actor) {
                            int true_face = (f >> actor) & 1;
                            int obs_face = X_2_ptr(t, actor);
                            double b_f = B_F_ptr(actor, true_face, obs_face);
                            if (b_f <= 0.0) {
                                obs_log_prob = -std::numeric_limits<double>::infinity();
                                break;
                            }
                            obs_log_prob += std::log(b_f);
                        }
                        
                        // 说话人观测
                        if (!std::isinf(obs_log_prob)) {
                            int obs_speaker = -1;
                            for (int s = 0; s < n_actors; ++s) {
                                if (X_1_ptr(t, s) == 1) {
                                    obs_speaker = s;
                                    break;
                                }
                            }
                            
                            if (obs_speaker >= 0) {
                                double b_s = B_S_ptr(rho, obs_speaker);
                                if (b_s > 0.0) {
                                    obs_log_prob += std::log(b_s);
                                } else {
                                    obs_log_prob = -std::numeric_limits<double>::infinity();
                                }
                            }
                        }
                        
                        if (!std::isinf(obs_log_prob)) {
                            U_ptr(t, f, rho) = max_trans_prob + obs_log_prob;
                        }
                    }
                }
            }
        }
    }
    
    return U;
}

// 嵌套HMM后向算法
py::array_t<double> nested_backward_pass(
    py::array_t<int> X_1,                // 说话人观测 [n_samples, n_actors]
    py::array_t<int> X_2,                // 面部出现 [n_samples, n_actors]
    py::array_t<double> alpha,           // 初始状态概率 [n_actors]
    py::array_t<double> A_F,             // 面部转移矩阵 [n_actors, 2, 2]
    py::array_t<double> beta,            // 说话人初始概率 [n_actors]
    py::array_t<double> A_S,             // 说话人转移矩阵 [n_actors, n_actors]
    py::array_t<double> B_F,             // 面部观测矩阵 [n_actors, 2, 2]
    py::array_t<double> B_S,             // 说话人观测矩阵 [n_actors, n_actors]
    double gamma1,
    double gamma2
) {
    auto X_1_ptr = X_1.template unchecked<2>();
    auto X_2_ptr = X_2.template unchecked<2>();  
    auto alpha_ptr = alpha.template unchecked<1>();
    auto A_F_ptr = A_F.template unchecked<3>();
    auto beta_ptr = beta.template unchecked<1>();
    auto A_S_ptr = A_S.template unchecked<2>();
    auto B_F_ptr = B_F.template unchecked<3>();
    auto B_S_ptr = B_S.template unchecked<2>();
    
    int T = X_1.shape(0);
    int n_actors = X_1.shape(1);
    int n_face_states = 1 << n_actors;
    
    // 后向变量 V[t, f, rho]
    auto V = py::array_t<double>({T, n_face_states, n_actors});
    auto V_ptr = V.template mutable_unchecked<3>();
    
    {
        py::gil_scoped_release nogil;
        
        // 初始化 V_{T-1}(f,ρ) = 0 (log domain)
        for (int f = 0; f < n_face_states; ++f) {
            for (int rho = 0; rho < n_actors; ++rho) {
                V_ptr(T-1, f, rho) = 0.0;
            }
        }
        
        // 后向递推 t = T-2, T-3, ..., 0
        for (int t = T-2; t >= 0; --t) {
            for (int f = 0; f < n_face_states; ++f) {
                for (int rho = 0; rho < n_actors; ++rho) {
                    std::vector<double> log_probs;
                    
                    for (int f_next = 0; f_next < n_face_states; ++f_next) {
                        for (int rho_next = 0; rho_next < n_actors; ++rho_next) {
                            double log_prob = V_ptr(t+1, f_next, rho_next);
                            
                            // P(F_{t+1,·} = f_next | F_{t,·} = f)
                            for (int actor = 0; actor < n_actors; ++actor) {
                                int curr_state = (f >> actor) & 1;
                                int next_state = (f_next >> actor) & 1;
                                log_prob += std::log(A_F_ptr(actor, curr_state, next_state));
                            }
                            
                            // P(S_{t+1} = rho_next | S_t = rho, F_{t+1,·} = f_next)
                            int rho_next_face_state = (f_next >> rho_next) & 1;
                            double unnorm_prob = A_S_ptr(rho, rho_next) * std::exp(gamma2 * rho_next_face_state);
                            
                            double normalizer = 0.0;
                            for (int r = 0; r < n_actors; ++r) {
                                int r_face_state = (f_next >> r) & 1;
                                normalizer += A_S_ptr(rho, r) * std::exp(gamma2 * r_face_state);
                            }
                            log_prob += std::log(unnorm_prob / normalizer);
                            
                            // 观测概率
                            for (int actor = 0; actor < n_actors; ++actor) {
                                int true_face = (f_next >> actor) & 1;
                                int obs_face = X_2_ptr(t+1, actor);
                                log_prob += std::log(B_F_ptr(actor, true_face, obs_face));
                            }
                            
                            // 说话人观测概率
                            int obs_speaker = 0;
                            for (int s = 0; s < n_actors; ++s) {
                                if (X_1_ptr(t+1, s) > 0.5) {
                                    obs_speaker = s;
                                    break;
                                }
                            }
                            log_prob += std::log(B_S_ptr(rho_next, obs_speaker));
                            
                            log_probs.push_back(log_prob);
                        }
                    }
                    
                    V_ptr(t, f, rho) = logsumexp(log_probs.data(), log_probs.size());
                }
            }
        }
    }
    
    return V;
}


// 添加Python模块绑定
PYBIND11_MODULE(nested_hmmc, m) {
    m.doc() = "Nested HMM C++ implementation";
    
    m.def("nested_forward_pass", &nested_forward_pass,
          "Nested HMM forward pass",
          py::arg("X_1"), py::arg("X_2"), py::arg("alpha"), py::arg("A_F"),
          py::arg("beta"), py::arg("A_S"), py::arg("B_F"), py::arg("B_S"),
          py::arg("gamma1"), py::arg("gamma2"));
    
    m.def("nested_backward_pass", &nested_backward_pass,
          "Nested HMM backward pass", 
          py::arg("X_1"), py::arg("X_2"), py::arg("alpha"), py::arg("A_F"),
          py::arg("beta"), py::arg("A_S"), py::arg("B_F"), py::arg("B_S"),
          py::arg("gamma1"), py::arg("gamma2"));
}