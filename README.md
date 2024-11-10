#### 2024_ia653 
## NLP Project: Tea Farkas & Lovro Banovic

MEETING 1 - 11/12/2024 - 1pm
</br>
#### PROJECT PROCESS:
**step 1:** Prepare the data by extracting recipes from cookbooks and saving them in a format suitable for the development of models.</br>
**step 2:** Determining the appropriate models for this data set and identifying those that will yield the most accurate predictions.</br>
**step 3:** Prepare the data for a specific model, construct the model, and assess its performance.</br>
**step 4:** In analyzing the results, the focus will be on graphing the data. The objective is to implement the most effective model within the Flask application.</br>
</br>

#### ABOUT THE DATA
Our data is collected from a few sources. One of these is a cookbook created by Tea, a member of our team, which will constitute a significant portion of our fitness recipe data. The second source is the website https://joyfoodsunshine.com/, which serves as our reference for regular recipes. As we continue gathering data, we plan to identify two additional sources—one for fitness recipes and another for regular recipes. This approach will enhance the versatility of our model. Our objective is to ensure that it can effectively process new and unseen data to the best possible extent.
</br>

![Dataset Sample](media/dataset.png)

We are organizing our data by compiling all regular recipes into a single list of dictionaries, referred to as *regular_food_recipes*, while all fitness recipes are collected in *fitness_food_recipes*. Subsequently, we will label these entries and merge them for classification purposes. Our goal is to be able to classify new recipes as either regular or fitness recipes.

### SIMPLE APPROACH: NAIVE BAYES MODEL
The Naive Bayes model is highly effective for text classification tasks.

**WHY?**
- Naive Bayes requires relatively small amounts of training data and is renowned for its simplicity, speed, and efficiency.


**WHY NOT?**
- The model's assumption of independence among features often does not hold true in real-world data. For example, in natural language processing tasks, certain words frequently appear together (such as "dark chocolate" or "zero sugar"), which poses a challenge for models that regard these occurrences as independent events.
</br>

![Predicted Class Probabilities for Test Recipes](media/naiveBayes_graph1.png)

### SIMPLE APPROACH: MULTINOMIAL NAIVE BAYES

The Multinomial Naive Bayes (MultinomialNB) classifier is particularly effective for text classification tasks.

**WHY?**  
This methodology excels with datasets that are represented by word frequencies, making it ideal for classifying recipes based on their ingredient lists, which have been transformed into TF-IDF (Term Frequency-Inverse Document Frequency) vectors. This approach allows the model to accurately reflect the importance of terms within the recipes, thereby aiding in the identification of patterns that distinguish "fitness" recipes from "regular" ones.

**WHY NOT?**  
However, the model may encounter challenges related to misclassification, especially when "fitness" and "regular" recipes share overlapping ingredients.
</br>
</br>
The classifier demonstrates an accuracy of 84%, exhibiting high precision for Fitness at 1.00, though it has a lower recall of 0.70. This indicates that some fitness recipes are being misclassified as regular. In contrast, for Regular, the precision stands at 0.75 with a perfect recall of 1.00. This means that all regular predictions are accurate, but there is still the possibility of misclassifying certain fitness recipes.
</br>

![Confusion Matrix for Recipe Classifier](media/confusionMatrix_graph2.png)

### COMPLEX APPROACH: NEURAL NETWORKS MODEL
...brainstorming...

### COMPLEX MODEL - ...
...brainstorming...

### FUTURE WORK  
In future developments, we will integrate the most effective model into the existing Flask application, enabling users to input their own meals and ascertain their alignment with fitness categories.
</br>

The adventure is just beginning... stay tuned for what's next!