# Targeted Marketing on Social Media: Utilizing Text Analysis to Create Personalized Landing Pages
This repository contains the code for the paper [Targeted Marketing on Social Media: Utilizing Text Analysis to Create Personalized Landing Pages](https://doi.org/10.21203/rs.3.rs-2728199/v1) by Yusuf M. Çetinkaya, Emre Külah, İsmail Hakkı Toroslu, and Hasan Davulcu. Sample dataset and the intermediate data i/o are provided in the [data](data) folder. The code is written in Python 3.9 and the notebooks are written in Jupyter Notebook.


![Pipeline](data/pipeline.png?raw=true "The pipeline for creating personalized landing pages")

The proposed approach comprises interconnected components that produce a customized, dynamically generated landing page. The pipeline steps are illustrated in Figure. The process begins with collecting tweets containing pre-determined keywords and filtering out unrelated ones. These relevant tweets are subsequently thematically modeled to extract priorities based on the codebook established by domain experts. Finally, topic probabilities are utilized to generate a coherent paragraph for the landing page that addresses the prospect's concerns.

## Notebooks
**Step1**: [Extracting Keywords from Facebook Posts](source/step1_keyword_detection.ipynb)

**Step2**: [Data Collection](source/step2_data_collection.ipynb)

**Step3**: [Analyzing Twitter Data with Embeddings and Clustering](source/step3_clustering_tweets.ipynb)

**Step4**: [Analyzing Twitter Data for Event-Related Clusters](source/step4_classifying_related_tweets.ipynb)

**Step5**: [Topic Modeling](source/step5_topic_modeling.ipynb)

**Step6**: [Landing Page Generation](source/step6_paragraph_generation.ipynb)

## Citation

If you use this code or find it useful in your research, please consider citing:

```
@article{ccetinkaya2024targeted,
  title={Targeted marketing on social media: Utilizing text analysis to create personalized landing pages},
  author={{\c{C}}etinkaya, Yusuf M{\"u}cahit and K{\"u}lah, Emre and Toroslu, {\.I}smail Hakk{\i} and Davulcu, Hasan},
  journal={Social Network Analysis and Mining},
  volume={14},
  number={1},
  pages={1--15},
  year={2024},
  publisher={Springer}
}

@inproceedings{ccetinkaya2022coherent,
  title={Coherent Personalized Paragraph Generation for a Successful Landing Page},
  author={{\c{C}}etinkaya, Yusuf M{\"u}cahit and Toroslu, {\.I}smail Hakk{\i} and Davulcu, Hasan},
  booktitle={2022 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM)},
  pages={252--255},
  year={2022},
  organization={IEEE}
}
```



