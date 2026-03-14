# test.py
import football_engine # This imports your C++ .so file!

print("Testing the C++ Football Oracle...")

# Call the C++ function directly from Python
result = football_engine.predict(2.5, 1.1)

if result == 1:
    print("Prediction: Home Team Wins!")
elif result == -1:
    print("Prediction: Away Team Wins!")
