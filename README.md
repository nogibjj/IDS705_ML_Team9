# IDS705_ML_Team9: Classification of Fake and Real Faces And Its Implications for Dating Apps

## Motivation
The problem of fake profiles on Dating Apps such as Tinder and Bumble is a serious issue that can deceive and scam users. The use of AI-generated photos to create fake profiles is a growing concern, with some individuals using them innocently due to disinterest or shyness, while others have malicious intentions. 
<img width="833" alt="Screen Shot 2023-04-19 at 8 35 40 PM" src="https://user-images.githubusercontent.com/60382493/233227857-b1f923dd-8d57-41db-a268-92dd7abe9fe7.png">

## Question:
1. What is the viability of applying our image classification model for fake and real face detection for a dating app company?

2. Will the model that we train misclassify faces due to some form of gendered or racial bias and/or issues with faces that do not have enough facial proximity or carry other types of features (such as wearing glasses)?

3. What are the limitations of both the model selection and the data of the application space?


## Data

The Flickr FacesHQ dataset contains 70,000 high-quality PNG images of human faces at 1024 x 1024 resolution, while the 1 million fake faces dataset contains artificially generated images. Both datasets exhibit variation in age, ethnicity. For our application space, it will be used to train and test a model for detecting fake images, with a focus on understanding the model's limitations and generalizability. The model will be tested on a typical dating profile faces to develop metrics for detecting fake user profiles. The dataset was manually curated to avoid legal and ethical concerns related to scraping profile images from dating apps. The goal is to understand the practicality and generalizability of future facial classification tools through these insights.


## Methods
### KNN (Baseline model)
<img width="677" alt="Screen Shot 2023-04-19 at 9 00 24 PM" src="https://user-images.githubusercontent.com/60382493/233230788-d8bcf0a4-4c06-4c27-b183-17c3075944aa.png">

### ResNet50: RGB
<img width="712" alt="Screen Shot 2023-04-19 at 9 00 05 PM" src="https://user-images.githubusercontent.com/60382493/233230763-f71dff3c-a9b7-42ab-b0bf-b3ec51e0a9ff.png">

### ResNet50: GrayScale
<img width="639" alt="Screen Shot 2023-04-19 at 8 59 45 PM" src="https://user-images.githubusercontent.com/60382493/233230723-cd3b687d-8253-4597-9f34-42e62fc33884.png">


## Results & Discussion



## Future Research
- Develop method for assessing impact of fake image quality on discriminative models
- Adversarial ML techniques can be used to create more sophisticated deepfakes to improve detection by training models on them
- Evaluate additional models such as ViT and CNN for comparison in interpretability and performance metrics
- Use unsupervised learning strategy (PCA) for feature extraction and identification of clusters or common points across different images
- Consider enlarging solution to 1024 x 1024 and use PCA to extract key features from high-dimensional data when feasible



## Reference
[1]  Olivera-La Rosa A, Arango-Tobón OE, Ingram GPD. Swiping right: face perception in the age of Tinder. Heliyon. 2019 Dec 2;5(12):e02949. doi: 10.1016/j.heliyon.2019.e02949. PMID: 31872122; PMCID: PMC6909076.

[2] Taeb, Maryam, and Hongmei Chi. 2022. "Comparison of Deepfake Detection Techniques through Deep Learning" Journal of Cybersecurity and Privacy 2, no. 1: 89-106. https://doi.org/10.3390/jcp2010007

[3] M. S. Rana, M. N. Nobi, B. Murali and A. H. Sung, "Deepfake Detection: A Systematic Literature Review," in IEEE Access, vol. 10, pp. 25494-25513, 2022, doi: 10.1109/ACCESS.2022.3154404.

[4] Shahzad HF, Rustam F, Flores ES, Luís Vidal Mazón J, de la Torre Diez I, Ashraf I. A Review of Image Processing Techniques for Deepfakes. Sensors (Basel). 2022 Jun 16;22(12):4556. doi: 10.3390/s22124556. PMID: 35746333; PMCID: PMC9230855.

[5] Maher Salman F, Abu-Naser S. Classification of Real and Fake Faces Using Deep Learning. 2022. International Journal of Academic and Engineering Research. Vol. 6, pp. 1-14. 2022 March. ISSN: 2643-9085. https://philarchive.org/archive/SALCOR-3

[6] Merrigan A, Smeaton A. Using a GAN to Generate Adversarial Examples to Facial Image Recognition. 2021 November 30. arXiv:2111.15213v. https://arxiv.org/pdf/2111.15213.pdf

[7] Wang, J., &amp; Li, Z. (2018). Research on face recognition based on CNN. IOP Conference Series: Earth and Environmental Science, 170, 032110. https://doi.org/10.1088/1755-1315/170/3/032110 

[8] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., &amp; Houlsby, N. (2021, June 3). An image is worth 16x16 words: Transformers for image recognition at scale. arXiv.org. Retrieved March 10, 2023, from https://arxiv.org/abs/2010.11929 

[9] M. Coşkun, A. Uçar, Ö. Yildirim and Y. Demir, "Face recognition based on convolutional neural network," 2017 International Conference on Modern Electrical and Energy Systems (MEES), Kremenchuk, Ukraine, 2017, pp. 376-379, doi: 10.1109/MEES.2017.8248937.

[10] Coccomini, D. A., Caldelli, R., Falchi, F., Gennaro, C., &amp; Amato, G. (2022, June 28). Cross-forgery analysis of vision transformers and CNNS for deepfake image detection. arXiv.org. Retrieved March 10, 2023, from https://arxiv.org/abs/2206.13829 

[11] Aura. The Unexpected Dangers of Online Dating [11 Scams to Know]. Retrieved March 10, 2023 from https://www.aura.com/learn/dangers-of-online-dating

[12] Datagen, "ResNet-50: The Architecture Explained," Datagen, Retrieved on April 5, 2023. [Online]. Available: https://datagen.tech/guides/computer-vision/resnet-50/

[13] Nvidia Corporation. (2019). FFHQ Dataset. [Online]. Available: https://github.com/NVlabs/ffhq-dataset. 

[14] S. Pumarola, A. Agudo, A. M. Martinez, A. Sanfeliu, F. Moreno-Noguer, "GANimation: One-Shot Anima-tion of Facial Expressions using Generative Adversarial Networks," arXiv preprint arXiv:1812.04948, 2018.

[15] V7 Labs. "1 Million Fake Faces." V7 Labs Open Datasets. https://www.v7labs.com/open-datasets/1-million-fake-faces. 

[16] Duke University Data Science Program, "Graduates of 2024," Accessed April 20, 2023. https://datascience.duke.edu/people/graduation-year/2024/. 

[17] Civitai. "Homepage," Accessed April 20, 2023. https://civitai.com/.

[18] Agarwal, R., AbdAlmageed, W., Wu, Y., & Natarajan, P. (2020). Deep Fake Detection: A Survey of Facial and Body Cues. In Proceedings of the 1st Workshop on Deep Learning for Deepfakes Detection (pp. 1-6). Papers with Code. Retrieved from https://paperswithcode.com/paper/deep-fake-detection-survey-of-facial 

[19] Ma, S., Wang, J., Huang, H., & Cui, Y. (2021). Deepfake Detection Based on Pupil Light Reflex and Convolutional Neural Networks. arXiv preprint arXiv:2106.14948.

[20] Federal Trade Commission. (2022, February). Reports of Romance Scams Hit Record Highs in 2021. [Online]. Available: https://www.ftc.gov/news-events/data-visualizations/data-spotlight/2022/02/reports-romance-scams-hit-record-highs-2021

[21] A. Smith and B. Johnson, "The Rise of Deepfake Technology: Implications for Society and National Security," IEEE Security & Privacy, vol. 18, no. 1, pp. 26-33, Jan.-Feb. 2020.

[22] J. Deng, W. Dong, R. Socher, L. Li, K. Li, and L. Fei-Fei, "ImageNet: A Large-Scale Hierarchical Image Database," ImageNet, Available: https://www.image-net.org/about.php. [Accessed: Apr. 19, 2023].

[23] N. N. Luong, T. Le, and D. Niyato, "A Comprehensive Survey on Blockchain Energy Systems," arXiv preprint arXiv:1909.11573, 2019. [Online]. Available: https://arxiv.org/pdf/1909.11573.pdf. [Accessed: Apr. 19, 2023].

[24] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," arXiv preprint arXiv:1512.03385v1, Dec. 2015. [Online]. Available: https://arxiv.org/abs/1512.03385v1. [Accessed: Apr. 19, 2023].

[25] N. Mir et al., "Fake Identity through Online Dating Applications," 2019 Curtin University's Networked Society Conference, Perth, Australia, 2019, pp. 1-5.

[26] P. Singh and P. Shukla, "Recognition of Fake Profiles in Social Media: A Literature Review," Gyan Vihar University Journals, vol. 1, no. 1, pp. 10-15, Aug. 2019.
[27] J. Smith and K. Lee, "The Role of Profile Pictures in Online Dating: A User Study," IEEE Transactions on Human-Machine Systems, vol. 49, no. 3, pp. 211-219, May 2019.

[28] C. Kanan and G. W. Cottrell, "Color-to-Grayscale: Does the Method Matter in Image Recognition?" in IEEE PLoS ONE, vol. 7, no. 1, pp. e29740, Jan. 2012, doi: 10.1371/journal.pone.0029740.

