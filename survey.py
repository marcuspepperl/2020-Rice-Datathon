def survey():
    f = open("Questions.txt", "r")
    questions = []
    f1 = f.read().splitlines()
    results = []

    print("Welcome to the marriage counselling survey! For each question in this survey, enter an integer between 0 and 4, inclusive. 0 is Almost Always, and 4 is Almost Never")
    for i in range (0, 54):
        answer = input(f1[i] + ' ')
        results.append(float(answer))
    return results

indices = survey()
