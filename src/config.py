import hyperopt as hpo


data_path = 'data'
models_path = 'models'
final_models_path = 'final_models'
final_pipeline_filename = 'final_pipeline.pkl'
final_model_filename = 'final_model.pkl'
seed = 42
target = 'revenue'
# top = 10
# threshold = 100
top = 50
threshold = 70

json_list_cols = {
    'genres': [
#         'id',
        'name',
    ],
    'production_companies': [
#         'id',
#         'logo_path',
        'name',
        'origin_country',
    ],
    'production_countries': [
        'iso_3166_1',
#         'name',
    ],
    'spoken_languages': [
        'iso_639_1',
#         'name',
    ],
    'Keywords': [
#         'id',
        'name',
    ],
    'cast': [
#         'cast_id',
#         'character',
#         'credit_id',
        'gender',
#         'id',
        'name',
#         'order',
#         'profile_path',
    ],
    'crew': [
#         'credit_id',
#         'department',
        'gender',
#         'id',
        'job',
        'name',
#         'profile_path',
    ],
}



one_hot_cols = [
    'original_language',
]

date_cols = [
    'release_date',
]

num_cols = [
    'budget',
    'popularity',
    'runtime',
    'video',
    'vote_average',
    'vote_count',
]

median_impute = [
    'runtime',
]

drop_cols = [
    'belongs_to_collection',
    'genres',
    'production_companies',
    'production_countries',
    'spoken_languages',
    'Keywords',
    'cast',
    'crew',
    'backdrop_path',
    'homepage',
    'id',
    'imdb_id',
    'original_title',
    'overview',
    'poster_path',
    'status',
    'tagline',
    'title',
    'release_date',
    'original_language',
    'revenue',
]

eval_cols = [
    'belongs_to_collection',
    'genres',
    'production_companies',
    'production_countries',
    'spoken_languages',
    'Keywords',
    'cast',
    'crew',
]


cols_select = [
    'vote_count',
    'budget',
    'popularity',
    'crew_len',
    'cast_len',
    'Keywords_len',
    'genres_len',
    'runtime',
    'Keywords_duringcreditsstinger',
    'genres_Action',
]

lgb_space = {
    'colsample_bytree':  hpo.hp.uniform('colsample_bytree', low=0.1, high=1.0),  # 1.0,
    'learning_rate':     hpo.hp.uniform('learning_rate', low=0.0, high=0.2),  # 0.1,
    'max_depth':         hpo.hp.quniform('max_depth', low=1, high=20, q=1),  # - 1,
    'min_child_samples': hpo.hp.quniform('min_child_samples', low=1, high=20, q=1),  # 20,
    'min_child_weight':  hpo.hp.uniform('min_child_weight', low=0, high=3),  # 0.001,
    'min_split_gain':    hpo.hp.uniform('min_split_gain', low=0, high=10),  # 0.0,
    'n_estimators':      hpo.hp.quniform('n_estimators', low=1000, high=3000, q=200),  # 100,
    'num_leaves':        hpo.hp.quniform('num_leaves', low=4, high=50, q=4),  # 31,
    'reg_alpha':         hpo.hp.uniform('reg_alpha', low=0.0, high=10.0),
    'reg_lambda':        hpo.hp.uniform('reg_lambda', low=0.0, high=10.0),
    'subsample':         hpo.hp.uniform('subsample', low=0.2, high=1.0),  # 1.0,
    'subsample_freq':    hpo.hp.quniform('subsample_freq', low=20, high=400, q=40),  # 0,
#     'subsample_for_bin': 200000,  # 200000,
#     'objective':         ,  # None,
#     'class_weight':      ,  # None,
#     'random_state':      ,  # None,
#     'n_jobs':            -1,  # -1,
#     'silent':            ,  # True,
#     'importance_type':   ,  # 'split',
}

xgb_space = {
    'base_score':              hpo.hp.uniform('base_score', low=0.6, high=1.0),
    'colsample_bylevel':       hpo.hp.uniform('colsample_bylevel', low=0.6, high=0.9),
    'colsample_bynode':        hpo.hp.uniform('colsample_bynode', low=0.8, high=1),
    'colsample_bytree':        hpo.hp.uniform('colsample_bytree', low=0.5, high=1),
    'gamma':                   hpo.hp.uniform('gamma', low=0, high=2),
    'learning_rate':           hpo.hp.uniform('learning_rate', low=0.0, high=0.1),
    'max_depth':               hpo.hp.quniform('max_depth', low=1, high=10, q=1),
    'min_child_weight':        hpo.hp.uniform('min_child_weight', low=1.0, high=30),
    'n_estimators':            hpo.hp.quniform('n_estimators', low=200, high=1000, q=100),
#     'max_delta_step':          hpo.hp.uniform('max_delta_step', low=0, high=1),
    'reg_alpha':               hpo.hp.uniform('reg_alpha', low=0, high=30),
    'reg_lambda':              hpo.hp.uniform('reg_lambda', low=0, high=10),
    'subsample':               hpo.hp.uniform('subsample', low=0.8, high=1.0),
}

rf_space = {
    'bootstrap':                hpo.hp.choice('bootstrap', [True, False]),  # True,
    'max_depth':                hpo.hp.choice('max_depth', [None, hpo.hp.quniform('max_depth_num', low=5, high=50, q=5), ]),  # - 1,
    'max_samples':              hpo.hp.choice('max_samples', [None, hpo.hp.uniform('max_samples_num', low=0.5, high=1), ]),  # None,
    'min_weight_fraction_leaf': hpo.hp.uniform('min_weight_fraction_leaf', low=0.0, high=0.5),  # 0.0,
    'n_estimators':             hpo.hp.quniform('n_estimators', low=50, high=500, q=50),  # 100,
#     'criterion':                hpo.hp.choice('criterion', ['mse', 'mae']),  # 'mse',
#     'min_samples_split':        hpo.hp.quniform('min_samples_split', low=2, high=100, q=10),  # 2,
#     'min_samples_leaf':         hpo.hp.quniform('min_samples_leaf', low=1, high=100, q=10),  # 1,
#     'max_features':             hpo.hp.choice('max_features', ['auto', 'sqrt', 'log2']),  # 'auto',   
#     'max_leaf_nodes':           ,  # None,
#     'min_impurity_decrease':    ,  # 0.0,
#     'oob_score':                ,  # False,
#     'n_jobs':                   ,  # None,
#     'random_state':             ,  # None,
#     'verbose':                  ,  # 0,
#     'warm_start':               ,  # False,
#     'ccp_alpha':                ,  # 0.0,
}


