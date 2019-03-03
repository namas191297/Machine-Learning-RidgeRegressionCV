import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
import random


''' Problem : We will generate a data-set containing the information regarding Squirrels. The information inclues size, weight and the intake of the squirrels
in grams. We need to use Ridge Regression to predict the size of the a mouse based on the other independent variables.
We will be implementing Ridge Regression using the linear Ridge model and using Cross-validation model.
We will be generating the values of the data-set ourselves and we will limit the dataset to minimum values as that is the whole
point of applying Ridge Regression Model, to predict efficiently even when there is limited data by using a Ridge Regression Penalty to
approximate the best fitting mapping function.

The linear regression equation can be initially considered to be :

SIZE = CONSTANT + 13.0*WEIGHT - 0.8*INTAKE

where CONSTANT = 12
'''

#Intializing parameters

CONSTANT = 12               # Also known as the Y-Intercept
TOTAL_SAMPLES = 200         # Total samples that are going to be used
MAX_SAMPLE_SIZE = 40        # Maximum size of a particular squirrel
INVOKE_ERROR = 0.02         # Error/Deviation that must be randomly applied to the data generated


#Defining functions to generate values for 13*WEIGHT and -0.8*INTAKE

def generateWeight(x):
    result = (x*13.0)
    deviation = 0
    deviation = result * random.uniform(-INVOKE_ERROR,INVOKE_ERROR)
    return(result + deviation)

def generateIntake(y):
    result = (y*(-0.80))
    deviation = 0
    deviation = result * random.uniform(-INVOKE_ERROR,INVOKE_ERROR)
    return (result + deviation)

#Generating values
random_array = np.random.randint(MAX_SAMPLE_SIZE,size=TOTAL_SAMPLES).astype(float)
weight = generateWeight(random_array)
intake = generateIntake(random_array)
rhs = weight + intake             #RHS of the given Equation
size = rhs + CONSTANT              #SIZE = RHS + Y-Intercept ===> SIZE = CONSTANT + 13.0*WEIGHT - 0.8*INTAKE
                                    #SIZES contains an array of the randomly generated values of the equation, as done in
                                    #Linear-Regression to fit a line.


#Preparing data for Ridge Regression with Cross Validation, Data is not partitioned into testing and training data and the complete data is fed into the model

weight_array = np.asarray(weight)
intake_array = np.asarray(intake)
size_array = np.asarray(size)

independent_dict = {'WEIGHT':weight_array,'INTAKE':intake_array}
dependent_dict = {'SIZE':size_array}

independent_df = pd.DataFrame(data=independent_dict)
dependent_df = pd.DataFrame(data=dependent_dict)

#Preparing the Cross-validation Ridge-regression model (Uses Leave-One-Out Cross-validation)

modelCV = RidgeCV(alphas=[1e-3,1e-2,1e-1,1],fit_intercept=True)
modelCV.fit(independent_df,dependent_df)

print("\nRidge-Regression model with cross validation predictions:",end="\n")
print("Prediction from a random value:",modelCV.predict([[random.uniform(weight.min(),weight.max()),random.uniform(intake.min(),intake.max())]]))
print("Model score:",modelCV.score(independent_df,dependent_df))
print("Co-efficients for line fitted by model after regression penalty:")
for coeff in modelCV.coef_:
    print(coeff)
print("Y-intercept for the model:",modelCV.intercept_)
print("\n\n")

