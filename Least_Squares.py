import math
import random
import numpy as np

def reduce_spacing(line):
    reduced_line = ''
    for ch in line:
        if ch.isdigit():
            reduced_line += ch + ' '
    return reduced_line



def bad_line(line):
    if len(line) == 0:
        return True
    for ch in line:
        if ch != ' ':
            return False
    return True

def read_matrix(filename):
    """
    Parse data from the file with the given filename into a matrix.

    input:
        - filename: a string representing the name of the file

    returns: a matrix containing the elements in the given file
    """


    fhandle = open(filename)
    all_data = []
    linenum = 0
    for line in fhandle:
        if linenum > 1:
            newline = reduce_spacing(line)
            newline.strip()
            att_data = newline.split()
            all_data.append(list(map(int, att_data)))
        linenum += 1

    new_data = []

    for entry in all_data:
        if entry != []:
            new_data.append(entry)

    data_matrix = np.matrix(new_data)

    return data_matrix


def testing_training(data_matrix, prop):

    nsamples, nattrs = np.shape(data_matrix)

    size_test_data = int(prop * nsamples)
    test_data_ind = random.sample(range(nsamples), size_test_data)
    training_data_ind = list(range(nsamples))

    for ind in test_data_ind:
        training_data_ind.remove(ind)

    training_data = data_matrix[training_data_ind, :-1]
    training_output = data_matrix[training_data_ind, -1]
    test_data = data_matrix[test_data_ind, :-1]
    test_output = put = data_matrix[test_data_ind, -1]


    return training_data, training_output, test_data, test_output

def generate_predictions_lst(weights, inputs):
    return sum([weight * input for weight, input in zip(weights, inputs)])

def generate_predictions(weights, inputs):

    return inputs @ weights

def prediction_error(weights, inputs, actual_result):
    # the predicted output is generated
    prediction = generate_predictions(weights, inputs)

    # the square error the vectors is computed
    square_error = 0
    vector_size = np.shape(prediction)[0]

    for ind in range(vector_size):

        square_error += (actual_result[ind, 0] - prediction[ind, 0]) ** 2

    # the mean square error is returned
    return prediction, square_error / vector_size


def fit_least_squares(input_data, output_data):

    # the weights that generate the minimum mse based on the input_data is
    # computed

    term1 = np.transpose(input_data) @ input_data
    term2 = np.transpose(input_data) @ output_data
    best_weights = np.linalg.inv(term1) @ term2

    # a linear model constructed from the fitted weights is returned
    return best_weights

def full_model(data_matrix):

    full_model_weights = fit_least_squares(data_matrix[:, :-1], data_matrix[:, -1]).tolist()

    return full_model_weights

def create_model(training_data, training_output, test_data, test_output):

    weights = fit_least_squares(training_data, training_output)
    prediction, error = prediction_error(weights, test_data, test_output)

    weights_vector = [entry[0] for entry in weights]

    indexed_weights = [tuple([count, entry]) for count, entry in enumerate(weights_vector)]

    return prediction, test_output, indexed_weights, error


def statistical_analysis(data_matrix):

    nsamples, nattrs = np.shape(data_matrix)
    ndivorced = sum(data_matrix[:, -1]).tolist()[0][0]
    nmarried = nsamples - ndivorced
    # print(ndivorced, nmarried)

    divorced_matrix = data_matrix[:ndivorced, :-1]
    married_matrix = data_matrix[ndivorced:, :-1]

    avg_divorced_attrs = []
    avg_married_attrs = []
    avg_difference_attrs = []

    for attr in range(nattrs - 1):
        attr_lst_1 = divorced_matrix[:, attr].tolist()
        attr_lst_1 = [entry[0] for entry in attr_lst_1]
        attr_lst_2 = married_matrix[:, attr].tolist()
        attr_lst_2 = [entry[0] for entry in attr_lst_2]

        avg_divorced_attrs.append(sum(attr_lst_1) / ndivorced)
        avg_married_attrs.append(sum(attr_lst_2) / nmarried)
        avg_difference_attrs.append(avg_divorced_attrs[-1] - avg_married_attrs[-1])


    full_model_weights = full_model(data_matrix)
    full_model_weights = [weight[0] for weight in full_model_weights]

    influence_divorced_attrs = [weight * divorced_score for weight, divorced_score in zip(full_model_weights, avg_divorced_attrs)]
    influence_married_attrs = [weight * married_score for weight, married_score in zip(full_model_weights, avg_married_attrs)]
    influence_difference_attrs = [weight * difference_score for weight, difference_score in zip(full_model_weights, avg_difference_attrs)]

    enumerate_divorced_attrs = [tuple([entry, count]) for count, entry in enumerate(influence_divorced_attrs)]
    sorted_influence_divorced_attrs = sorted(enumerate_divorced_attrs, reverse = True)

    enumerate_married_attrs = [tuple([entry, count]) for count, entry in enumerate(influence_married_attrs)]
    sorted_married_divorced_attrs = sorted(enumerate_married_attrs, reverse = True)

    enumerate_difference_attrs = [tuple([entry, count]) for count, entry in enumerate(influence_difference_attrs)]
    sorted_influence_difference_attrs = sorted(enumerate_difference_attrs, reverse = True)

    print('Most Significant Attributes in General:')
    for entry in sorted_influence_difference_attrs:
        print('Attribute:', entry[1], 'Weight:', entry[0])

    print('\n')




def scenario_analysis(model_weights, attr_scores, categories):

    divorcity_index = generate_predictions_lst(model_weights, attr_scores)
    print('Your divorcity index is', divorcity_index, '\n')
    nattrs = len(attr_scores)
    category_impact_lst = []

    for category, attrs in categories.items():
        category_impact = 0
        for attr in attrs:
            category_impact += model_weights[attr] * attr_scores[attr]
        category_impact_lst.append(tuple([category_impact / len(attrs), category]))


    category_impact_lst.sort(reverse = True)

    print('\nIn order of most critical factors:\n')
    for entry in category_impact_lst:
        print(entry[1], entry[0])


def survey(model_weights, categories):
    f = open("Questions.txt", "r")
    questions = []
    f1 = f.read().splitlines()
    results = []

    print("Welcome to the marriage counselling survey! For each question in this survey, enter an integer between 0 and 4, inclusive. 0 is Almost Always, and 4 is Almost Never")
    for i in range (0, 54):
        answer = input(f1[i] + ' ')
        results.append(float(answer))
    scenario_analysis(model_weights, results, categories)




def run_experiment(filename):

    #define categories
    categories = {'optimism' : tuple([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    'shared values' : tuple([9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
    'understanding' : tuple([20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 38, 39]),
    'stonewalling' : tuple([41, 42, 43, 44, 45, 46, 47, 48, 49, 50]),
    'criticism' : tuple([30, 31, 32, 33, 34, 35, 36, 37, 40, 51, 52, 53])}

    data_matrix = read_matrix(filename)
    nsamples, nattrs = np.shape(data_matrix)
    nattrs -= 1

    prop = 1 / 8
    total_error = 0
    total_indexed_weights = list(enumerate([0] * 54))
    trials = 10

    for count in range(trials):
        training_data, training_output, test_data, test_output = testing_training(data_matrix, prop)
        prediction, test_output, indexed_weights, error = create_model(training_data, training_output, test_data, test_output)
        for num in range(nattrs):
            total_indexed_weights[num] = tuple([num, total_indexed_weights[num][1] + indexed_weights[num][1]])
        total_error += error

    avg_error = total_error / trials

    print('\nThe average predictive error of a Linear Model trained using a random subset of the data over', trials, 'trials is:', avg_error, '\n')

    statistical_analysis(data_matrix)

    full_model_weights = full_model(data_matrix)
    full_model_weights = [weight[0] for weight in full_model_weights]

    survey(full_model_weights, categories)




run_experiment('divorce.txt')
