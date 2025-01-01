## Related Work + Criticism

#### Automatic Mushroom Species Classification Model for Foodborne Disease Prevention Based on Vision Transformer
- [Journal](https://www.researchgate.net/publication/362770084_Automatic_Mushroom_Species_Classification_Model_for_Foodborne_Disease_Prevention_Based_on_Vision_Transformer)
- [Local PDF](./papers/Automatic_Mushroom_Species_Classification_Model_fo.pdf)
- Criticism: Overestimation of Performance:
    - Data based Flaws:
        - This paper aims to classify images of 12 different species of mushrooms and gets reasonably good results. Chosen mushroom species are visually distinct, which makes the classification task easier. The authors should have included more visually similar species to test the model's robustness and generalization capabilities.


#### Machine Learning and Image Processing-Based System for Identifying Mushrooms Species in Malaysia
- [Journal](https://www.mdpi.com/2076-3417/14/15/6794)
- [Local PDF](./papers/applsci-14-06794.pdf)
- Criticism: Good Results but contains Flaws:
    - Methodological Flaws: 
        - uniformly chosen hyperparameters
        - Using the same hyperparameter set across different architectures (e.g., CNNs and Vision Transformers) is flawed because optimal hyperparameters differ based on the model's design and training dynamics. This approach biases results toward the model that aligns better with the arbitrarily chosen parameters which leads to an unfair comparison not representative of the models' true capabilities. Optimally, one would perform hyperparameter tuning for each model separately to ensure a fair comparison and use the best hyperparameters for each model for the final evaluation.

    - Misnomer/Problematic Phrasing: 
        - "Additionally, this study also compares the performance of the traditional convolutional neural network architecture against a novel transformer architecture with the same task, under the same parameters. The experiment shows promising results and proves that deep learning can be used to identify different species of mushrooms using an image-based dataset."
        - Misnomer as CNNs are not distinct from deep learning, so supposedly CNNs should have proven already that deep learning can be used to identify different species of mushrooms using an image-based dataset.


#### Mushroom Classification using CNN and Gradient Boosting Models
- [Journal](https://ieeexplore.ieee.org/abstract/document/10689875)
- [Local PDF](./papers/Mushroom_Classification_Using_CNN_and_Gradient_Boosting_Models.pdf)
- Criticism: Overstates the Applicability of CNNs and GBMs for (mushroom) classification:
    - Overstatement:
        - "Thus, the feature proves the effectiveness of CNNs
        and GB in precise classification of mushrooms for their further
        application in a variety of spheres"
        - Dataset consists of only 3 classes. Achieving good results (i.e. high acc/f1/...) is easier with fewer classes as decision boundaries are simpler. 


#### Deep Learning-Based Classification of Macrofungi: Comparative Analysis of Advanced Models for Accurate Fungi Identification
- [Journal](https://www.mdpi.com/1424-8220/24/22/7189)
- [Local PDF](./papers/sensors-24-07189.pdf)
- Investigate: Their findings indicate Transformers to be less effective than Dense CNNs for mushroom classification. Maybe DCNNs perform better when the dataset is comprised of fewer classes, as is the case in this study (only 6 classes)?


### Novelty of Our Work
- Significantly larger dataset with currently 43 classes
- Similar performance to state-of-the-art models on smaller datasets even with uneven class distribution
- Related work shows that misclassification is still a problem which reinforces our decision to include the top k predictions as a decision support system for the user rather than taking the top prediction as the final decision.

### Future Work
- on-device port (tensorflow lite)? questionable performancewise as transformers are computationally expensive
- Information Retrieval: Implement map showing reported mushroom locations in Austria / Used for human decision support.



## Experimentss and Results
<!-- conducted experiments, findings -->
Initially, three different models were trained/finetuned on the augmented training dataset. 



## Conclusions
<!-- key takeaways, future work -->