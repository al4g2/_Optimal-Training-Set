def feature_list(features):
    if features == '1':
        feature_cols = ['TAT']
    elif features == '2':
        feature_cols = ['TAT', 'Laser Power']
    elif features == '3':
        feature_cols = ['TAT', 'MaxTemp', 'CoolingRate']  # three original feature columns
    elif features == '4':
        feature_cols = ['TAT', 'MaxTemp', 'CoolingRate', 'Laser Power']
    elif features == '5':
        feature_cols = ['TAT', 'MaxTemp', 'CoolingRate', 'Z', 'Laser Power']
    elif features == '12':
        feature_cols = ['TAT', 'MaxTemp', 'CoolingRate',
                        'Median TAT 1', 'Median Tp 1', 'Median CR 1',
                        'Median TAT 2', 'Median Tp 2', 'Median CR 2',
                        'Median TAT 3', 'Median Tp 3', 'Median CR 3']  # 12 features: 3 + 9 medians
    elif features == '13':
        feature_cols = ['TAT', 'MaxTemp', 'CoolingRate', 'Laser Power',
                        'Median TAT 1', 'Median Tp 1', 'Median CR 1',
                        'Median TAT 2', 'Median Tp 2', 'Median CR 2',
                        'Median TAT 3', 'Median Tp 3', 'Median CR 3']  # 13 features: 12 + Laser Power
    elif features == '19':
        feature_cols = ['TAT',
                        'Median TAT 1', 'Median Tp 1', 'Median CR 1',
                        'Median TAT 2', 'Median Tp 2', 'Median CR 2',
                        'Median TAT 3', 'Median Tp 3', 'Median CR 3',
                        'Binomial Blur TAT', 'Black Top Hat TAT', 'Bounded Reciprocal TAT',
                        'Box Mean TAT', 'H Convex TAT', 'H Maxima TAT',
                        'H Minima TAT', 'Log10 TAT', 'White Top Hat TAT']  # TAT features and Laser Power
    elif features == '23':
        feature_cols = ['TAT', 'MaxTemp', 'CoolingRate', 'Z', 'Laser Power',
                        'Median TAT 1', 'Median Tp 1', 'Median CR 1',
                        'Median TAT 2', 'Median Tp 2', 'Median CR 2',
                        'Median TAT 3', 'Median Tp 3', 'Median CR 3',
                        'Binomial Blur TAT', 'Black Top Hat TAT', 'Bounded Reciprocal TAT',
                        'Box Mean TAT', 'H Convex TAT', 'H Maxima TAT',
                        'H Minima TAT', 'Log10 TAT', 'White Top Hat TAT']
    elif features == '39':
        feature_cols = ['TAT', 'MaxTemp', 'CoolingRate',
                        'Median TAT 1', 'Median Tp 1', 'Median CR 1',
                        'Median TAT 2', 'Median Tp 2', 'Median CR 2',
                        'Median TAT 3', 'Median Tp 3', 'Median CR 3',
                        'Binomial Blur TAT', 'Black Top Hat TAT', 'Bounded Reciprocal TAT',
                        'Box Mean TAT', 'H Convex TAT', 'H Maxima TAT',
                        'H Minima TAT', 'Log10 TAT', 'White Top Hat TAT',
                        'Binomial Blur Tp', 'Black Top Hat Tp', 'Bounded Reciprocal Tp',
                        'Box Mean Tp', 'H Convex Tp', 'H Maxima Tp',
                        'H Minima Tp', 'Log10 Tp', 'White Top Hat Tp',
                        'Binomial Blur CR', 'Black Top Hat CR', 'Bounded Reciprocal CR',
                        'Box Mean CR', 'H Convex CR', 'H Maxima CR',
                        'H Minima CR', 'Log10 CR', 'White Top Hat CR']  # 39 features: 12 + 27 ITK features
    elif features == '40':
        feature_cols = ['TAT', 'MaxTemp', 'CoolingRate', 'Laser Power',
                        'Median TAT 1', 'Median Tp 1', 'Median CR 1',
                        'Median TAT 2', 'Median Tp 2', 'Median CR 2',
                        'Median TAT 3', 'Median Tp 3', 'Median CR 3',
                        'Binomial Blur TAT', 'Black Top Hat TAT', 'Bounded Reciprocal TAT',
                        'Box Mean TAT', 'H Convex TAT', 'H Maxima TAT',
                        'H Minima TAT', 'Log10 TAT', 'White Top Hat TAT',
                        'Binomial Blur Tp', 'Black Top Hat Tp', 'Bounded Reciprocal Tp',
                        'Box Mean Tp', 'H Convex Tp', 'H Maxima Tp',
                        'H Minima Tp', 'Log10 Tp', 'White Top Hat Tp',
                        'Binomial Blur CR', 'Black Top Hat CR', 'Bounded Reciprocal CR',
                        'Box Mean CR', 'H Convex CR', 'H Maxima CR',
                        'H Minima CR', 'Log10 CR', 'White Top Hat CR']  # 40 features: 39 + Laser Power
    elif features == '41':
        feature_cols = ['TAT', 'MaxTemp', 'CoolingRate', 'Z', 'Laser Power',
                        'Median TAT 1', 'Median Tp 1', 'Median CR 1',
                        'Median TAT 2', 'Median Tp 2', 'Median CR 2',
                        'Median TAT 3', 'Median Tp 3', 'Median CR 3',
                        'Binomial Blur TAT', 'Black Top Hat TAT', 'Bounded Reciprocal TAT',
                        'Box Mean TAT', 'H Convex TAT', 'H Maxima TAT',
                        'H Minima TAT', 'Log10 TAT', 'White Top Hat TAT',
                        'Binomial Blur Tp', 'Black Top Hat Tp', 'Bounded Reciprocal Tp',
                        'Box Mean Tp', 'H Convex Tp', 'H Maxima Tp',
                        'H Minima Tp', 'Log10 Tp', 'White Top Hat Tp',
                        'Binomial Blur CR', 'Black Top Hat CR', 'Bounded Reciprocal CR',
                        'Box Mean CR', 'H Convex CR', 'H Maxima CR',
                        'H Minima CR', 'Log10 CR', 'White Top Hat CR']  # 41 features: 39 + Z + Laser Power
    else: feature_cols = ['error']

    return feature_cols

# features = '3'
# feature_cols = list(features)
# print(feature_cols)
