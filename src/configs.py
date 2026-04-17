class _C:
    # 基础配置
    model_name = "FCMIL"
    seed = 42
    gpus = "0"

    # 路径配置
    data_root = "/path/to/wsi_features"
    splits = "/path/to/dataset_splits"
    log_folder = "./logs"
    pretrained_path = None


    n_class = 2
    epochs = 100
    lr = 3e-4
    train_batch_size = 1
    test_batch_size = 1
    num_workers = 4


    alpha_cr = 0.1
    topk_ratio = 0.03
    lam = 0.5

    # 数据集交叉验证
    folds = "0,1,2,3,4"
    data_inst = "camelyon16"  # 用于日志命名


    label_list = ["Normal", "Tumor"]
    mapping = "0:0, 1:1"