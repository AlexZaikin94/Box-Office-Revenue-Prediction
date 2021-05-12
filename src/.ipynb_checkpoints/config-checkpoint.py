data_path = 'data'
models_path = 'models'
seed = 42
target = 'revenue'
top = 50
threshold = 30

json_list_cols = {
    'genres': 'name',
    'production_companies': 'name',
    'production_countries': 'name',
    'spoken_languages': 'iso_639_1',
    'Keywords': 'name',
    'crew': 'name',
    'cast': 'name',
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
    'original_language'
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




