# Topic Modeling by Aspect Extraction Algorithm

For topic modeling, we adopted an [unsupervised Attention-based Aspect Extraction algorithm](https://www.aclweb.org/anthology/P17-1036.pdf), which is a fancy attention-based adaption of low rank matrix approximation. 

The goal is to reconstruct aspect embeddings matrix from Attention-based contextual word embedding.
They use this newly constructed aspect embeddings to translate sentences, such that every sentence would be a linear interpolation of L aspect embeddings.
During training, they try to minimize the differences between the interpolation and representation by novel Attention based embedding. 
The Resulting aspect embeddings would be the desired output. 
The initialization of aspect embeddings adopts K-means clustering results, so the results are pretty similar to K-means results with Attention-based contextual word embedding. Please refer to the original paper for further details.

We adopted the [python3 implementation](https://github.com/harpaj/Unsupervised-Aspect-Extraction) forked from 
[original codes](https://github.com/ruidan/Unsupervised-Aspect-Extraction).

## Results

num_aspect is set to 14, which outputs equally weighted 14 clusters of topics. Yet topics are sorted by weights in each topic. Results after 5 epochs:

Aspect 0: parties, celebrations 

Aspect 1: nature, transportation

Aspect 2: internet, social networking thing

Aspect 3: food 

Aspect 4: politic correctness, current issues, LGBTQ 

Aspect 5: education, job related, linkedin stuff

Aspect 6: outfit and fashion, appearance and sexual related

Aspect 7: this aspect is a bit too broad, it contains only verbs.

Aspect 8: health issue

Aspect 9: chores, housework related

Aspect 10: money issue

Aspect 11: marriage, family and relationship.

Aspect 12: furniture. This might be in the same category as aspect 9.

Aspect 13: offensive words… related to religion, race, lgbt.

We could catch a glimpse of what people are discussing. Note that not every aspect makes sense. The last two seem to repeat previous aspects. So this is just a rough sketch of topics people talked about. Maybe in future interpretation tasks, we could identify aspect of each submission, and observe how our ethics bot performs within different aspects.
