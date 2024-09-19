Predicting Overall Athlete Wellness and Injury Risk in Collegiate Tennis Players: A Machine
Learning Approach

FRANCISCO ERRAMUSPE ALVAREZ, Department of Computer Science and Software Engineering,
Monmouth University, USA

SHOBHARANI POLASA, Department of Computer Science and Software Engineering, Monmouth
University, USA

WEIHAO QU, Department of Computer Science and Software Engineering, Monmouth University, USA

JAY WANG, Department of Computer Science and Software Engineering, Monmouth University, USA

The rapid development of machine learning has positively impacted every aspect of life, including the field of
sports. This study investigates the application of machine learning (ML) to predict overall athlete wellness and
potential injury risk in collegiate tennis players, aiming to assist coaches and trainers in identifying athletes at
risk and implementing personalized interventions to enhance performance and prevent injuries. To this end,
data from nine collegiate tennis players, encompassing physiological, training, sleep, and self-reported data,
were preprocessed and used to train and evaluate statistical models including Linear Regression, XGBoost
(Regressor and Classifier), Logistic Regression, Decision Tree, Random Forest; and deep learning models such
as MLP and LSTM. The results demonstrate the potential of XGBoost in achieving accurate predictions of
overall wellness and classifying injury risk, achieving an R-squared of 0.8378 for wellness prediction and an
AUC-ROC of 0.6947 for injury risk classification. Despite promising results, the limited sample size calls for
further data collection and model refinement to enhance generalizability and clinical applicability. WQ: TODO:
add some about deep learning results, and the comparison between XGBoost..
Additional Key Words and Phrases: Athlete Wellness, Injury Prevention, Machine Learning, Wearable Tech-
nology, Collegiate Tennis, XGBoost, Linear Regression, Logistic Regression, Decision Tree, Random Forest

ACM Reference Format:
Francisco Erramuspe Alvarez, Shobharani Polasa, Weihao Qu, and Jay Wang. 2024. Predicting Overall Athlete
Wellness and Injury Risk in Collegiate Tennis Players: A Machine Learning Approach. 1, 1 (September 2024),
17 pages. https://doi.org/10.1145/nnnnnnn.nnnnnnn

1 INTRODUCTION
Optimizing athletic performance while minimizing the risk of injury is crucial in professional sports.
Achieving this balance requires a comprehensive understanding of the diverse factors influencing
an athlete’s well-being, including physiological, psychological, training, and lifestyle factors. The
rapid development of machine learning has greatly promoted the development of performance and
injury prediction in professional sports. Recent works showed some key factors such as workout
schedule, player characteristics, injury history, and workload could affect injury risk. However,
these approaches rely heavily on subjective observations and expert assessments [6], which can be
limited in their ability to capture the complex interplay of these factors.
Authors’ addresses: Francisco Erramuspe Alvarez, s1365567@monmouth.edu, Department of Computer Science and Software
Engineering, Monmouth University, 400 Cedar Avenue, West Long Branch, New Jersey, 07764, USA; Shobharani Polasa,
s1365603@monmouth.edu, Department of Computer Science and Software Engineering, Monmouth University, 400 Cedar
Avenue, West Long Branch, New Jersey, 07764, USA; Weihao Qu, wqu@monmouth.edu, Department of Computer Science
and Software Engineering, Monmouth University, 400 Cedar Avenue, West Long Branch, New Jersey, 07764, USA; Jay Wang,
jwang@monmouth.edu, Department of Computer Science and Software Engineering, Monmouth University, 400 Cedar
Avenue, West Long Branch, New Jersey, 07764, USA.
© 2024 Copyright held by the owner/author(s). Publication rights licensed to ACM.

This is the author’s version of the work. It is posted here for your personal use. Not for redistribution. The definitive Version
of Record was published in , To overcome the shortness of only relying on subjective observation data in recent machine
learning-based performance prediction and injury production models, The advent of wearable
technology has influenced every aspect of life, from detecting potential falls [13, 14] and alerting
users to abnormal heart rates to analyzing sleep quality [ 17] and tracking workout progress. On
the other hand, the growing use of athlete self-report measures (ASRM) has demonstrated its
potential to enhance athletic performance [18 ]. The marriage of wearable technology and athlete
surveys has ushered in a new era of data-driven approaches to athlete management. The advantage
of combining wearable technology with athlete surveys is the ability to gather a wealth of both
objective information such as an athlete’s physiological responses, training loads, and sleep patterns,
and subjective insights, including feelings, stress, pain, and fatigue. This comprehensive approach
offers a more complete understanding of an athlete’s condition, enabling the development of more
objective and personalized interventions.

This study explores the potential of machine learning (ML) to leverage this data and predict
overall athlete wellness, as well as the risk of potential injuries. While contact sports like football,
soccer, and rugby tend to have the highest injury rates, non-contact sports such as track and field,
swimming, and tennis also experience injuries, often due to overuse and repetitive motion [1]. In
particular, a significant number of casual tennis players suffer from injuries caused by incorrect
playing techniques, with well-known examples including tennis elbow and tennis knee [7 ]. This
study specifically focuses on collegiate tennis players and employs a variety of ML algorithms to
develop a predictive model that integrates physiological data, workout details, sleep information,
and self-reported survey responses. This study has the potential to offer personalized guidance for
casual and professional tennis players, helping to protect them from injuries.
TODO: Combine the following to the above. This model offers a two-pronged approach to athlete
management, predicting both their physical capability and risk of injury. It achieves this through
two interconnected models that share core physiological and training load data as input. The
first model, a regression model, focuses on predicting an athlete’s current physical capability. By
analyzing physiological indicators like heart rate variability and sleep quality, alongside training
load data such as activity strain and jump height, the model generates a percentage score reflecting
the athlete’s overall condition. This score, for example, 90% indicating excellent condition and 60%
suggesting a need for adjusted training, allows for personalized training plans and performance
optimization. Simultaneously, the second model, a classification model, predicts the likelihood
of an athlete sustaining an injury within a specific timeframe. This model incorporates the same
physiological and training data as the first, but also considers the athlete’s injury history, including
type, duration, and severity of past injuries. The output is a percentage representing the probability
of injury, with a 50% cut-off used to classify athletes as either "High Risk" or "Low Risk." For
instance, a 90% prediction signifies a very high risk, necessitating immediate intervention, while
20% suggests relatively low risk. The real power of this approach lies in the interconnectedness
of the models. The athlete’s physical capability, as predicted by the regression model, directly
impacts their injury risk. A lower physical capability, potentially due to fatigue or inadequate
recovery, can significantly increase the susceptibility to injury. Therefore, by combining insights
from both models, coaches and trainers gain a comprehensive understanding of an athlete’s well-
being. This allows for informed decisions, such as modifying training intensity or implementing
load management strategies, to optimize performance while mitigating injury risk.

2 RELATED WORK

The application of machine learning (ML) in sports analytics has earned significant attention in
recent years, particularly in predicting athlete performance and injury risk. Several studies have Predicting Overall Athlete Wellness and Injury Risk in Collegiate Tennis Players: A Machine Learning

Approach 3

explored various ML methodologies to help enhance athletic performance and mitigate injury risks,
providing a foundation upon which the current study builds.
Claudino et al. [2 ] provided a systematic review of ML applications in sports injury prediction
and prevention, noting the predominance of supervised learning approaches like Artificial Neural
Networks (ANN), Decision Trees (DT), and Support Vector Machines (SVM). Their review high-
lighted the effectiveness of ensemble methods in improving prediction accuracy and handling
complex, high-dimensional datasets. Similarly, Van Eetvelde et al. [ 23] emphasized the role of ML
in identifying injury risk factors and developing predictive models that assist in proactive injury
prevention strategies. Overall, related studies focus on considering two factors: Machine learning
models and Data.

2.1 Tennis

One work studies the association and predictive ability of several markers of internal workload on
risk of injury in high-performance junior tennis players and shows that a high acute workload is
only one of the many factors associated with injury, and by itself, has low predictive ability for
injury [ 10]. Another work deployed machine learning techniques to develop a predictive model
capable of estimating an individual’s probability of getting injured in the future based on features
such as player characteristics, injury histories, and recent tournament schedules [9]. The possible
relationship of age, maturation, and the physical components; “upper body power”, “lower body
power”, “speed”, and “agility” with tennis performance at U13 and U16 was analyzed by Regression
analyses [8 ]. Sipko and Knottenbelt [19] used a supervised machine learning approach that uses
historical player performance across a wide variety of statistics to predict match outcomes, including
logistic regression and artificial neural networks evaluated on a test set of 6315 ATP matches played
in one year. According to Sampaio et al. [16], machine learning shows promise in psychological
state monitoring, talent identification, match outcome prediction, spatial and tactical analysis, and
injury prevention. Machine learning applications Coaches can leverage wearable technologies for
personalized psychological state monitoring, data-driven talent identification, and tactical insights
for informed decision-making. Gescheit [5] studies the injury facts among elite tennis players by
examining the epidemiology and in-event treatment frequency of injury at the 2011-2016 Australian
Open tournaments. The data includes sex, injury region and type and is reported as frequencies
per 10,000 game exposures. However, the sensor data collected from the wearable devices during
the players’ regular training, match might reveal more constructive inner information about the
players than just statistics data. The investigation [ 11] indicates an acute increase in load was
associated with increased injury risk after the analysis of the workload and self-reported injuries
in junior tennis players.

2.2 Machine Learning Models

Several studies have focused on college baseball athletes, utilizing various machine learning models
to predict both performance and the likelihood of injury. These insights reinforce the rationale
behind our choice of ML models, such as XGBoost, known for its ability to handle diverse feature
sets and capture intricate patterns within the data; and neural network architectures such as
MLP [21] and LSTM [24].
XGBoost Based Performance and Injury Prediction. Taber et al . [20] conducted a holistic evaluation
of player-, team-, and conference-level performances in Division-1 Women’s basketball using ML
techniques. Their study employed data from training, subjective stress, sleep, recovery metrics
(collected via WHOOP straps), in-game statistics, and countermovement jumps. Utilizing Extreme
Gradient Boosting (XGB) classifiers and regressors, they achieved high accuracy and F1 scores n predicting key performance indicators such as the Reactive Strength Index modified (RSImod),
game scores, and Player Efficiency Rating (PER). The ensemble approach, incorporating Random
Forest and correlation analyses, underscored the importance of various features across different
performance levels. This multi-tiered methodology aligns closely with our approach, which also
integrates physiological, training, sleep, and self-reported data to predict overall athlete wellness and
injury risk. However, while Taber et al. [20] focused on performance metrics, our study extends this
framework to specifically address injury risk classification alongside wellness prediction, thereby
offering a more comprehensive tool for athlete management. Another study focused on injury
assessment examined sixteen young female basketball players in China using various machine
learning models, demonstrating the strong performance of XGBoost in this area [6 ]. The difference
between our work and [6 ] is that their study relies on self-reported data from athletes, without
utilizing wearable technologies. Additionally, there are notable differences in injury prediction
between contact sports like basketball and non-contact sports like tennis.
Neural Network Based Injury Prediction. Still focusing on college basketball athletes, Zhao [25]
explored the use of neural networks to predict basketball injuries among college athletes. The study
compared traditional Backpropagation (BP) neural networks, Scaled Conjugate Gradient (SCG)
neural networks, and Radial Basis Function (RBF) neural networks, finding that RBF networks
achieved the highest prediction accuracy of 95.4%. Zhao emphasized the importance of selecting
appropriate ML algorithms to handle the complexities of injury data, highlighting the superior
performance of RBF networks in capturing nonlinear relationships inherent in injury predictors.
Our research similarly employs advanced ML models, including XGBoost and neural networks
(MLP and LSTM), to predict injury risks. By leveraging ensemble methods and deep learning
architectures, our study aims to enhance prediction accuracy and model robustness, addressing
some of the limitations noted by Zhao [25], such as the generalizability of models across different
sports and genders.

2.3 Data Choice

The selection of data is also crucial in accurately predicting performance and assessing injury risks.
Integration of Wearable Technology and Self-Reported Data. The integration of wearable technol-
ogy data with self-reported metrics has been a recurring theme in sports analytics research. It has
been reported that self-reported data can effectively provide new insights into the interactions
between competition-related stressors experienced by professional athletes and their susceptibility
to illness [22]. The wearable fitness band, Whoop, has been widely adopted in college athletics,
including sports such as softball, women’s lacrosse, baseball [15], and wrestling [ 4 ]. In the college
basketball studies [20, 25 ], the value of combining objective physiological data with subjective mea-
sures such as stress and recovery questionnaires is revealed to enhance the predictive power of ML
models. Our study similarly amalgamates data from wearable devices, workout logs, sleep patterns,
and athlete surveys to create a multifaceted dataset. This comprehensive data integration facilitates
a more nuanced understanding of athlete wellness and injury risk, enabling the development of
personalized intervention strategies.

Time-Series Data. Recent advancements in deep learning, particularly with models like Multi-
Layer Perceptrons (MLP) and Long Short-Term Memory (LSTM) networks, have shown promise
in capturing temporal dependencies and complex nonlinear relationships in time-series data. In
sports like basketball [20], volleyball [3 ], tennis, and soccer [ 12 ], data is typically collected over
seasons, aligning with the characteristics of time series data. Our implementation of MLP and
LSTM models aims to leverage these strengths, enabling the prediction of an athlete’s physical Predicting Overall Athlete Wellness and Injury Risk in Collegiate Tennis 

Players: A Machine Learning

Approach 5
capability and readiness based on their physiological and activity metrics over time. This approach
not only complements traditional ML models but also addresses the temporal dynamics that are
critical in understanding athlete performance and injury risk.
Feature Importance and Explainability. Understanding feature importance and ensuring model
explainability are crucial for the practical adoption of ML models in sports settings. Taber et
al. [20] utilized Partial Dependence Plots (PDPs) and ensemble feature importance techniques to
elucidate the impact of various features on performance metrics. Similarly, our study employs
feature importance analyses to identify the most significant predictors of athlete wellness and injury
risk, thereby providing actionable insights to coaches and trainers. This emphasis on explainability
aligns with the broader trend in ML research towards transparent and interpretable models, which
are essential for gaining trust and facilitating informed decision-making in sports management.

2.4 Limitations and Future Directions
Despite the advancements, existing studies often grapple with limitations such as small sample sizes,
data imbalance, and the generalizability of models across different populations and sports [ 20, 25 ].
Our study acknowledges these challenges and addresses them through robust data preprocessing,
feature engineering, and the implementation of advanced ML techniques. Future research can build
upon our findings by expanding the dataset, incorporating more diverse injury types, and exploring
federated learning approaches to enhance model scalability and privacy.
In summary, the current body of literature underscores the transformative potential of ML in
sports analytics, particularly in performance prediction and injury prevention. Our study contributes
to this growing field by developing a comprehensive ML framework that integrates diverse data
sources to predict overall athlete wellness and injury risk in collegiate tennis players. By building
on the methodologies and insights from prior research, we aim to advance the application of ML in
creating safer and more effective training programs for athletes.


3 METHODS
3.1 Participants
Data were collected from nine collegiate tennis players (Male: 5, Female: 4; Age: 20.3 ± 1.5 years;
Height: 175.2 ± 8.7 cm; Weight: 68.9 ± 9.2 kg) over a period of 16 weeks. All participants were
informed about the study procedures and provided written consent.
3.2 Data Collection
Data were gathered from five distinct sources, each providing unique and complementary informa-
tion essential for comprehensive analysis.
Journal Entries were collected through daily self-reported responses using online questionnaires
administered via Google Forms. These entries encompassed details on alcohol consumption, caffeine
intake, calorie tracking, and areas of discomfort such as knees, ankles, and other joints. This
qualitative data was pivotal in understanding factors that might affect performance and injury risk,
offering insights into both lifestyle habits and potential musculoskeletal issues.
Physiological Metrics were obtained using WHOOP wearable devices, which provided continu-
ous monitoring of various physiological parameters. The collected metrics included Recovery Score
(%), Resting Heart Rate (bpm), Heart Rate Variability (ms), Skin Temperature (°C), Blood Oxygen
Saturation (%), Day Strain, Energy Burned (kcal), Maximum Heart Rate (bpm), Average Heart
Rate (bpm), and detailed Sleep Metrics. The utilization of WHOOP devices allowed for objective,
high-resolution physiological data collection, which is critical for monitoring athlete readiness and
recovery n addition to the device-generated data, Sleep Data was supplemented with subjective as-
sessments of sleep quality and any sleep disturbances not captured by the WHOOP devices. This
information was gathered via daily questionnaires, providing a comprehensive view of the athletes’
sleep patterns and their impact on overall performance and recovery.
Workout Data included detailed information about the athletes’ training sessions, encompassing
tennis practices, matches, and strength and conditioning workouts. Data points such as duration,
intensity, and type of each workout were meticulously recorded through collaboration with coaching
staff and the use of training logs. This data was instrumental in understanding the physical demands
placed on athletes and their potential impact on recovery and performance.
Vertical Jump Data measured the athletes’ explosive lower-body power through a standardized
testing protocol conducted weekly or bi-weekly. Athletes performed three countermovement jumps,
and the average jump height (in inches) was recorded. This metric provided valuable insights into
neuromuscular readiness and fatigue levels, contributing to the assessment of overall athletic
performance.
Collecting data from these five sources enabled a multidimensional analysis of athlete wellness,
performance, and injury risk. The integration of subjective self-reports, objective physiological
measurements, sleep assessments, workout details, and performance metrics facilitated a compre-
hensive understanding of each athlete’s condition. This holistic approach is essential for developing
personalized interventions and enhancing predictive modeling.

3.3 System Architecture
Figure ?? illustrates the overall architecture of our athlete wellness and injury risk prediction
system. This framework integrates data from various sources, processes it through multiple stages,
and utilizes different machine learning models to generate predictions and actionable insights.
The system begins with data collection from multiple sources, including WHOOP wearable
devices, self-reported questionnaires, and vertical jump tests. This data undergoes cleaning and
preprocessing, followed by feature engineering to extract meaningful attributes. The processed data
is then fed into three main predictive models: Physical Capability Prediction (using MLP/LSTM),
Injury Risk Classification (using XGBoost Classifier), and Overall Wellness Prediction (using XG-
Boost Regressor). The outputs from these models are combined to calculate an Athlete Readiness
Score (ARS), which provides a comprehensive assessment of the athlete’s condition. This score is
then used to generate personalized recovery advice through a language model, offering tailored
recommendations to athletes and coaches.

3.4 Data Preprocessing
The collected data were preprocessed using Python’s Pandas library and other data processing
tools. The following steps were performed to ensure data quality and suitability for model training:
(1) Data Cleaning: Placeholder values such as ’N/A’, ’Unknown’, ’–’, empty strings, and spaces
were replaced with NaN to standardize missing values across the dataset.
(2) Data Type Conversion: Columns were converted to appropriate data types. Numeric
columns stored as objects due to formatting issues were converted to numerical data types
using coercion, which replaced non-convertible values with NaN.
(3) Missing Value Imputation: Missing numerical values were imputed using the mean of
each column to maintain consistency without introducing bias. For categorical variables,
missing values were imputed using the mode (most frequent value). redicting Overall Athlete Wellness and Injury Risk in Collegiate Tennis Players: A Machine Learning
Approach 7

(a) Sample Questionnaire (b) Player Wearing WHOOP Device
(c) Sample WHOOP Data (d) Jump Data Collection Setup
Fig. 1. Data Collection Methods: (a) Sample Questionnaire, (b) Tennis Player Aryna Sabalenka Wearing a
WHOOP Device During a Match [Photo Credit: USA TODAY / VIA REUTERS], (c) Sample WHOOP Data, (d)
Jump Data Collection Setup. Image source for (b): https://www.japantimes.co.jp/sports/2024/08/14/tennis/
aryna-sabalenka-short-memory/
(4) Creation of ’Injury Risk’ Target Variable: Injury-related columns were identified by
searching for keywords such as ’pain’, ’discomfort’, ’injury’, ’aches’, and ’stiffness’ in the
column names. The column:
"Did you experience today any type of pain or discomfort that did not allow you to play
tennis/perform daily activities properly? If so, please describe the area."
was selected and renamed to Injury_Description for simplicity. A new binary target
variable, Injury Risk, was created, where:
• 1 indicates the athlete reported any pain or discomfort affecting performance.
• 0 indicates no pain or discomfort reported.
Variations in responses (e.g., ’No’, ’none’, ’N/A’) were accounted for to accurately categorize
injury risk.
(5) Feature Selection: Relevant features were selected based on domain knowledge and their
potential predictive power for the target variables, Recovery Score (%) and Injury Risk.
Features with little to no variance were excluded to improve model performance.
(6) Handling Multicollinearity: Variance Inflation Factor (VIF) analysis was conducted to
identify multicollinearity among features. Features with high VIF values were considered
for removal or transformation to reduce redundancy and improve model interpretability.Feature Scaling: Numerical features were standardized using the StandardScaler to ensure
that each feature contributed equally to the model training process.
This comprehensive preprocessing approach addressed potential issues such as missing values,
inconsistent data types, and multicollinearity, resulting in a clean and reliable dataset for modeling.
3.5 Feature Engineering
Building upon the initial feature set, additional features were engineered to enhance the model’s
ability to capture complex patterns.
Total Sleep Duration was calculated by summing "Asleep duration (min)" and "Awake duration
(min)", providing a comprehensive measure of sleep over a period. This feature offers a holistic
view of an athlete’s sleep patterns, essential for assessing overall recovery and readiness.
Sleep Debt was determined by subtracting "Asleep duration (min)" from "Sleep need (min)".
This metric indicates whether an athlete is meeting their sleep requirements, highlighting potential
deficits that could impact performance and injury risk.
To address skewed distributions and satisfy linear regression assumptions, Logarithmic Trans-
formations were applied to variables such as "Heart Rate Variability (ms)". This normalization
step ensures that the data conforms more closely to a normal distribution, enhancing the model’s
predictive accuracy.
Interaction Terms were created between key variables like "Activity Strain" and "Asleep
duration (min)". These interaction features capture the combined effect of multiple factors on
recovery, allowing the model to detect synergistic influences that might not be apparent when
considering variables in isolation.
Polynomial Features were generated to enable the model to capture non-linear relationships
between predictors and the target variable. By introducing squared and cubic terms of existing
features, the model gains the flexibility to fit more complex patterns within the data.
Finally, Categorical Variable Encoding was performed on categorical variables such as "Activity
Name". Utilizing one-hot encoding, these categorical features were converted into a numerical
format, making them suitable for inclusion in machine learning models. This encoding preserves the
distinct categories without imposing any ordinal relationship, ensuring that the model interprets
each category appropriately.
These engineered features aimed to provide the model with richer information, improving its
predictive capability and addressing potential non-linearities in the data. By incorporating domain
knowledge into feature creation, we enhanced the model’s ability to detect subtle patterns related
to athlete wellness and performance.
3.6 Machine Learning Models
3.6.1 Overall Wellness Prediction (Regression). To predict the athletes’ Recovery Score (%), a variety
of linear and ensemble regression methods were employed, each contributing uniquely to the
predictive performance of the model.
Baseline Linear Regression was initially utilized to establish a performance benchmark,
operating under the assumption of a linear relationship between the independent variables and
the target variable. This fundamental approach provided a reference point against which more
complex models could be evaluated.
Enhancing the linearity and adhering to model assumptions, Linear Regression with Feature
Transformation involved applying logarithmic transformations to skewed features such as "Heart
Rate Variability (ms)". This normalization step was crucial in ensuring that the data distribution
conformed more closely to a normal distribution, thereby improving the model’s predictive accuracy. Predicting Overall Athlete Wellness and Injury Risk in Collegiate Tennis Players: A Machine Learning
Approach 9
To capture the combined effects of multiple variables on recovery scores, Linear Regression
with Interaction Terms was implemented. By introducing interaction terms between key features
like "Activity Strain" and "Asleep duration (min)", the model could detect synergistic influences
that might not be apparent when considering variables independently.
Recognizing the presence of non-linear relationships within the data, Polynomial Regression
was employed. This approach involved generating polynomial features, allowing the model to
fit more complex patterns and enhancing its ability to capture non-linear dependencies between
predictors and the target variable.
Addressing potential overfitting and improving feature selection, Lasso Regression (L1 Regu-
larization) was applied. Lasso regression penalizes the absolute size of the coefficients, effectively
shrinking less important feature coefficients toward zero. This regularization technique not only
prevents overfitting but also aids in identifying the most influential predictors by performing
feature selection.
Finally, XGBoost Regressor, an ensemble learning method based on gradient boosting, was
utilized. XGBoost builds multiple weak predictive models sequentially, each correcting the errors of
its predecessor, and combines them to form a powerful ensemble model. Renowned for its efficiency
and ability to handle complex non-linear relationships, XGBoost significantly enhanced the model’s
predictive capabilities by leveraging the strengths of gradient boosting techniques.
These models were systematically trained and evaluated to determine the most effective approach
for predicting recovery scores. Emphasizing advanced feature engineering and leveraging ensemble
methods, the chosen models demonstrated improved performance, providing robust predictions of
athlete wellness and facilitating informed decision-making for personalized interventions.
3.6.2 Injury Risk Classification. In addition to regression models, various classification algorithms
were employed to predict injury risk based on the collected data.
Logistic Regression was utilized as a foundational statistical model, employing a logistic
function to model a binary dependent variable. This model is particularly suitable for understanding
the impact of various factors on injury risk, providing insights into the probability of injury
occurrence based on predictor variables.
Enhancing predictive accuracy and addressing data imbalance, the XGBoost Classifier was
implemented. As an ensemble learning method that builds multiple classification trees using
gradient boosting, XGBoost effectively handles complex non-linear relationships and interactions
between features. Its robustness makes it adept at managing imbalanced datasets, which is crucial
given the typically lower incidence of injuries compared to non-injuries.
The Decision Tree Classifier was employed for its interpretability and simplicity. This tree-
structured model splits the data into subsets based on the value of input features, making it easy to
visualize and understand the decision-making process behind injury risk predictions.
To further improve model performance and control overfitting, the Random Forest Classifier
was used. As an ensemble of decision trees, Random Forest aggregates the predictions of multiple
trees, enhancing accuracy and providing greater stability compared to individual decision trees.
This method leverages the diversity of trees to minimize variance and improve generalization to
unseen data.
Recognizing the inherent Class Imbalance in injury occurrence data, strategies such as adjusting
class weights and employing stratified sampling were integrated during model training. These
techniques aim to enhance the models’ ability to accurately predict injury occurrences by ensuring
that the minority class (injuries) is adequately represented and learned by the classifiers. These classification models collectively aimed to identify athletes at higher risk of injury by
learning patterns and associations within the data. By leveraging both linear and ensemble meth-
ods, the models were able to capture a wide range of relationships and interactions, providing a
comprehensive framework for injury risk prediction.
3.6.3 MLP. The Multi-Layer Perceptron (MLP) model was trained to predict the athlete’s physical
capability percentage based on a sequence of 7 days of physiological and activity metrics. The
model architecture consisted of fully connected layers with ReLU activations and a final sigmoid
layer to output a value between 0 and 100.
The model was trained for 100 epochs on a dataset of 82,804 sequences. The training process
showed a general decrease in both training and validation loss, with the final epoch reporting a
training loss of 0.0388 and a validation loss of 0.6391.
To interpret the MLP output, we use the following formula:
Physical Capability Score = MLP_output ∗ 100 (1)
This scales the model’s output to a percentage between 0 and 100, representing the athlete’s
predicted physical capability. Higher scores indicate better readiness for performance.
The model’s predictions on the test set ranged from approximately 58 to 80 percent physical
capability, providing a nuanced view of athlete readiness across different days and conditions.
3.6.4 LSTM. The Long Short-Term Memory (LSTM) model was designed to capture temporal
dependencies in the athlete data, using a sequence of 7 days to predict physical capability. The
LSTM architecture included 64 hidden units and 2 layers, followed by a fully connected layer and a
sigmoid activation.
The LSTM model was trained for 100 epochs on the same dataset of 82,804 sequences. The final
epoch showed a training loss of 0.0360 and a validation loss of 0.6059, indicating slightly better
performance than the MLP model.
The interpretation of the LSTM output uses the same formula as the MLP:
Physical Capability Score = LSTM_output ∗ 100 (2)
The LSTM model’s predictions on the test set also ranged from approximately 58 to 80 percent
physical capability, similar to the MLP results. However, the LSTM model may capture more
nuanced temporal patterns in the data, potentially providing more accurate predictions over time.
3.7 Model Evaluation Metrics
The following evaluation metrics were employed to assess the performance of the developed models.
Regression Metrics were utilized to evaluate the accuracy and reliability of the models predict-
ing continuous outcomes, such as the athletes’ Recovery Score (%). Mean Absolute Error (MAE)
measures the average magnitude of errors between the predicted and actual values, providing a
straightforward assessment of prediction accuracy without considering the direction of errors.
Root Mean Squared Error (RMSE) assesses the model’s prediction accuracy by penalizing larger
errors more than smaller ones, thereby highlighting significant discrepancies between predicted
and actual values. R-squared (R2 Score) represents the proportion of variance in the dependent
variable that is predictable from the independent variables, indicating the model’s explanatory
power and how well the predictors account for the variability in the target variable.
Classification Metrics were employed to evaluate the performance of models predicting cate-
gorical outcomes, such as injury risk classification. Accuracy denotes the proportion of correct
predictions out of the total number of predictions made, offering a basic measure of overall mode Predicting Overall Athlete Wellness and Injury Risk in Collegiate Tennis Players: A Machine Learning
Approach 11
performance. F1 Score is the harmonic mean of precision and recall, providing a balanced measure
that is particularly important in imbalanced datasets where one class may be underrepresented.
AUC-ROC (Area Under the Receiver Operating Characteristic Curve) measures the ability of
the model to distinguish between classes, with a higher AUC indicating better discriminatory
power. Confusion Matrix offers a detailed breakdown of the model’s performance by displaying
true positives, false positives, true negatives, and false negatives, facilitating a comprehensive
understanding of prediction errors and model behavior.
Additionally, residual analysis was performed for regression models to validate the assumptions
underlying linear regression. This analysis included assessing the linearity, homoscedasticity,
independence, and normality of residuals to ensure the robustness and reliability of the regression
models.

3.8 Code Availability
The code for data preprocessing, model training, and evaluation is available in a public GitHub
repository at https://github.com/franciscoerramuspe/masters_thesis. The repository includes all
scripts necessary to replicate the results discussed in this paper, as well as detailed instructions for
setting up the environment and running the experiments.

4 RESULTS
4.1 Overall Wellness Prediction
We evaluated multiple regression models for predicting the athletes’ Recovery Score (%). The
performance metrics for these models are presented in Table 1.
Table 1. Regression Model Performance for Overall Wellness Prediction
Model MAE RMSE R2 Score
Baseline Linear Regression 14.05 16.84 0.009
Linear Regression with Feature Transformation 11.32 14.27 0.215
Linear Regression with Interaction Terms 10.47 13.58 0.310
Polynomial Regression 8.23 11.40 0.537
Lasso Regression 9.76 12.85 0.402
XGBoost Regressor 3.82 6.81 0.838

4.1.1 Model Performance Comparison. The Polynomial Regression model demonstrated signifi-
cant improvement over the baseline linear regression model, with an R2 score of 0.537. However,
the XGBoost Regressor achieved the best performance overall, indicating its superior ability to
capture complex non-linear relationships in the data.

4.1.2 Predicted vs. Actual Values. Figure 2 illustrates the relationship between the actual and
predicted recovery scores using the Polynomial Regression model. The scatter plot shows that
the predicted values closely align with the actual values, as many data points are near the diagonal
line representing perfect predictions.

4.1.3 Feature Importance. The XGBoost model’s feature importance analysis identified the most
significant predictors of recovery scores:
• Heart Rate Variability (ms)
• Resting Heart Rate (bpm)
• Activity Strain The confusion matrix shows that while the model performs well in identifying non-injury cases, it
has limitations in correctly predicting injury occurrences, which is critical for preventive measures.
4.2.3 Handling Class Imbalance. To address class imbalance:
• Adjusting Class Weights: The XGBoost Classifier was configured with a scale_pos_weight
parameter to give more weight to the minority class.
• Stratified Sampling: Ensured that the training and testing sets maintained the same class
distribution.
• Consideration for Future Work: Techniques such as oversampling the minority class (e.g.,
SMOTE) or using specialized algorithms for imbalanced data were identified as potential
strategies to improve model performance.

5 DISCUSSION OF RESULTS

The optimization of regression models and the application of ensemble methods have yielded
substantial improvements in predictive performance, particularly for overall wellness prediction.
However, injury risk classification remains challenging due to class imbalance and limited sample
size.

5.1 Overall Wellness Prediction
The XGBoost Regressor significantly outperformed all linear regression models, achieving an
R2 score of 0.838 compared to 0.537 for the best-performing Polynomial Regression model. This ndicates that XGBoost is highly effective in capturing the complex non-linear interactions within
the physiological and performance data, making it a robust tool for predicting athlete wellness.
Key Insights:
• Feature Importance: Heart Rate Variability and Resting Heart Rate emerged as top predic-
tors, aligning with physiological theories that link these metrics to recovery and readiness.
• Model Robustness: The high R2 score suggests that the model can reliably explain a
significant portion of the variance in recovery scores, which is valuable for making informed
decisions in athlete management.

5.2 Injury Risk Classification
While the XGBoost Classifier showed better performance compared to other models, the overall
low F1 Scores highlight persistent challenges in accurately predicting injury risks. This is likely
due to the small sample size and the inherent difficulty in predicting rare events.

Key Insights:

• Class Imbalance: The disparity between injury and non-injury cases adversely affects
model performance, particularly in predicting the minority class.
• Model Sensitivity: Enhancing the model’s sensitivity to injury risk is crucial for developing
effective preventive strategies.
5.3 Correlation Between Physical Capability and Injury Probability
An essential aspect of injury prevention is understanding how an athlete’s physical capabilities
correlate with their likelihood of sustaining injuries. Our analysis reveals a significant correlation
between specific physiological metrics and injury risk:
• Heart Rate Variability (HRV): Lower HRV indicates higher stress levels and inadequate
recovery, which are associated with an increased probability of injuries.
• Resting Heart Rate (RHR): Elevated RHR can be a sign of fatigue or overtraining, both of
which heighten injury risk.
• Activity Strain: Higher activity strain correlates with greater physical stress, potentially
leading to overuse injuries if not managed properly.
These correlations underscore the importance of monitoring physiological indicators to proac-
tively identify athletes at risk of injury. By integrating these metrics into predictive models, coaches
and trainers can implement targeted interventions to mitigate injury risks.
5.4 Upper Body and Lower Body Injury Prediction
In addition to overall injury risk, our study delves into predicting specific injury categories, namely
upper body and lower body injuries. This differentiation allows for more precise interventions
tailored to the athlete’s needs.
Labeling Methodology: To facilitate this, injuries were categorized based on the reported area
of discomfort:
• Upper Body Injuries: Includes injuries to the arms, shoulders, and upper torso, such as
tennis elbow and shoulder strains.
• Lower Body Injuries: Encompasses injuries to the legs, knees, and lower torso, including
tennis knee and hamstring strains.
Each injury type was labeled accordingly in the dataset, enabling the application of XGBoost
classifiers to predict the likelihood of specific injury categories.
XGBoost Application: Using the labeled data, separate XGBoost models were trained for upper
body and lower body injury predictions. These models demonstrated improved specificity, allowing Predicting Overall Athlete Wellness and Injury Risk in Collegiate Tennis Players: A Machine Learning

Approach 15
for targeted preventive measures. For instance, identifying a higher risk of upper body injuries
can prompt interventions like strength training and technique adjustments, while a propensity for
lower body injuries may lead to focus on flexibility and endurance training.

6 FUTURE WORK

Building on the findings of this study, several avenues for future research are proposed:
• Expanding the Sample Size: Increasing the number of participants will improve model
robustness and generalizability.
• Incorporating Detailed Injury Data: Collecting comprehensive injury records, including
type and severity, will enhance prediction accuracy.
• Addressing Data Imbalance: Implementing techniques like SMOTE or using specialized
algorithms can improve injury risk classification.
• Integrating Real-Time Data: Utilizing continuous data streams from wearable devices
can enable real-time monitoring and timely interventions.
• Developing an Injury Advising System: Creating a system that provides personalized
recommendations based on model predictions can translate insights into actionable strate-
gies.
• Conducting User Studies: Engaging with athletes and coaches to assess the practicality
and effectiveness of the predictive tools will provide valuable feedback.
• Leveraging Advanced Machine Learning Techniques: Exploring deep learning models,
such as LSTM networks, to capture temporal dependencies in the data.
• Implementing Federated Learning: Utilizing federated learning can allow for data shar-
ing across institutions while maintaining privacy, thereby enhancing model performance
without compromising sensitive information.
• Predicting Specific Injury Sites: Future work will focus on developing models that predict
injuries in specific body parts, such as the knee and left arm. This targeted approach will
enable more precise preventive measures and interventions tailored to the unique injury
profiles of athletes.
7 CONCLUSION
This research demonstrates the promising potential of machine learning, particularly ensemble
methods like XGBoost, for athlete wellness prediction and injury risk classification in collegiate
tennis.
• Key Findings:
– Advanced models outperformed traditional linear models in capturing complex rela-
tionships.
– Physiological metrics collected from wearable devices were valuable predictors of
recovery and injury risk.
– Handling class imbalance is critical for improving injury risk classification.
• Implications:
– Coaches and trainers can leverage such models for proactive athlete management.
– Personalized interventions can be developed to enhance performance and prevent
injuries.
However, further investigation is needed, focusing on increasing the sample size, collecting
detailed injury data, and exploring additional features that contribute to athlete well-being. Address-
ing data imbalance and enhancing model generalizability are essential steps toward developing
reliable predictive tools.
