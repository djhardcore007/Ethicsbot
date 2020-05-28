For topic modeling, I adopted an unsupervised Attention-based Aspect Extraction algorithm. 
In my understanding, this algorithm is just a fancy adaption of low rank matrix approximation. 
The ultimate goal is to reconstruct aspect embeddings matrix $T$ (with $L$-dimension) from Attention-based contextual word embedding $E$ ($V$-dimension, $V$ = vocab_size), where $L << V$. 
They use this newly constructed T to translate sentences, such that every sentence would be a linear interpolation of $T$ (representing L aspects). 
During training, they try to minimize the difference between this interpolation and representation by original Attention based embedding. 
Resulting T would be desired aspects. 
According to the paper, the initialization of $T$ adopts K-means clustering results, 
so the results are pretty similar to K-means results with Attention-based contextual word embedding. 
For further details, see original paper: \url{https://www.aclweb.org/anthology/P17-1036.pdf}

This new unsupervised topic modeling algorithm adopts the popular Attention mechanism and is able to yield far better results than LDA. 
The reason I prefer this model to LDA is: 

1) it adopts Attention based contextual word embedding, which is consistent with our Bert based binary classifier; 

2) it exceled LDA in every possible aspect according to the paper.

I used this python3 implementation (https://github.com/harpaj/Unsupervised-Aspect-Extraction) forked from 
original codes by the paper author (https://github.com/ruidan/Unsupervised-Aspect-Extraction).

I arbitrarily set parameter num_aspect = 14, which outputs 14 clusters of topics, which are supposed to be equally weighted. Yet topics are sorted by weights in each aspect. The outputs after 5 epochs are saved  here. 

From this, we could catch a glimpse of what people are discussing. Note that not every aspect makes sense. The last two seem to repeat previous aspects. So this is just a rough sketch of topics people talked about…

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

Aspect 12: furniture. This might be in the same category as aspect 9. This means that we could reduce num_aspect = 14.

Aspect 13: offensive words… related to religion, race, lgbt… This might be in the same category as aspect 4. 

\textbf{Suggestions}:

Maybe in future interpretation tasks, we could identify aspect of each submission, and observe how our ethics bot perform within different aspects.
