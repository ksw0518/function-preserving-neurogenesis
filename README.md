# function-preserving-neurogenesis
바보들아
멍청이들아

# config.json 주석
CONFIG = {
    'methods': ['standard'],#원래 'standard', 'cbp', 'rejuvenation', 'ours', 'ours_no_noise'
    'seeds': [0], #원래 0,1,2
    'num_tasks': 2,          # Permuted MNIST 과제 수 (계획서: 수백~수천, 일단 작게 시작)
    'epochs_per_task': 2,
    'lr': 0.001,
    'p': 0.05,                # 하위 p% 제거 (1, 5, 10%)
    'noise_std': 0.01,        # 대칭 깨기 노이즈 표준편차
    'beta': 0.99,             # utility EMA 계수
    'maturity': 100,          # 이 step 이상 산 뉴런만 제거 후보
    'k': 50,                  # 재생성 전후 비교 구간 (step)
    'near_dead_threshold' : 0.95
}