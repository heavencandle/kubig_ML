# Machine Learning
- based on "Hands-On Machine Learning with Scikit-Learn and TensorFlow by Aurélien Géron"

## 0. Library Spec
- scikit-learn v.0.22.1
## 1. Machine Learning Basics
### 1-1. What is Machine Learnning
1. Machine Learning
    1. field of study that gives computers the ability to learn without being explicitly programmed.
    2. A coputer program i said to learn from experience E with respect to some task T and some performance measure P, if it performance on T, as measured by P, improves with experience E.

2. Machine Learning is great for:
    1. Problems for which exiting solutions require a lot of hand-tuning or long lists of rules
    2. Complex problems for which there i no good solution at all using a traditional approach 
    3. Fluctuating environments (i.e. can adapt to new data)
    4. Getting insight about complex problems and large amounts of data

3. Machine Learning Classification
    1. supervised / unsupervised
    2. batch learning(offline) / online learning
    3. instance based / model based
        - instance based: learns the example by heart(memorize), generalizes to new cases based on similarity  
        - model based: learns model, decides based on model  
  
### 1-2. Model Tuning and Measurements
1. parameters - learning rate
    - learning rate= how fast model hould adapt to change
    - high: rapidly adapt to new data, but tend to quickly forget the old data 
    - low: have more inertia. learn more slowly, less sensitive to noise

2. Measurements: utility function / cost funciton
    - utility function(fitness function): how **good** the model is
    - cost function: how **bad** the model is
    
3. train, validation, test set
    - generalization error: error rate on new case
    - **model selection: select the model and hyperparameters that perform best on the validation set.** After final model is selected, train hyperparameters on the full training set and measure generalized error on the test set
    - cross validation: the training set is split into complementary subsets, and each model i trained against a different combination of subsets and validated against the remaining parts. (_e.g. splitted training set A, B, C. train w/ A&B and validate on C. train w/ A&C and validate on B and so on_)

### 1-3. Possible Problems
1. Data driven problems
    1. insufficiency
    2. nonrepresentative(e.g. sampling bias)
    3. poor-quality
    4. irrelevant features ---> feature selection / feature extraction
 
2. Overfitting : the model performs well on the training data, but does not generalize well(=underperform on the test data). **happens when the models is too complex relative to the amount and noisiness of the training data.**
    - solution
    ```
    1)simplify the model by selecting one with fewer prarmeters, reduce number of attributes
      (e.g. a liear model rather than a high-degree polynomial model)
    2)gather more training data   
    3)reduce the noise in the training data(e.g. fix data errors,dand remove outliers)
    ```

3. underfitting: when the model is too simple to learn the structure of data
    - solution
    ```
    1)select more complex model with more parameters  
    2)add better features  
    3)reduce the constraints(e.g. regularization)
    ```  
						 
  

