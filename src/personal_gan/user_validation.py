def user_validation(prompt):
    while True:
        response = input(prompt).lower()
        if response in ['y', 'n']:
            return response
        else:
            print("Invalid input. Please enter 'y' for yes or 'n' for no.")