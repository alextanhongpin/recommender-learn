````python name=bayesian_personalized_ranking.py
import numpy as np
import random

class BPR:
    def __init__(self, num_users, num_items, embedding_dim, learning_rate=0.01, reg=0.01):
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.reg = reg
        self.user_embeddings = np.random.normal(size=(num_users, embedding_dim))
        self.item_embeddings = np.random.normal(size=(num_items, embedding_dim))

    def train(self, user_item_pairs, num_epochs=10):
        for epoch in range(num_epochs):
            for user, pos_item in user_item_pairs:
                neg_item = self.sample_negative_item(user, pos_item)
                self.update(user, pos_item, neg_item)

    def sample_negative_item(self, user, pos_item):
        neg_item = pos_item
        while neg_item == pos_item:
            neg_item = random.randint(0, self.num_items - 1)
        return neg_item

    def update(self, user, pos_item, neg_item):
        user_embedding = self.user_embeddings[user]
        pos_item_embedding = self.item_embeddings[pos_item]
        neg_item_embedding = self.item_embeddings[neg_item]

        x_uij = np.dot(user_embedding, pos_item_embedding) - np.dot(user_embedding, neg_item_embedding)
        sigmoid = 1 / (1 + np.exp(x_uij))

        user_embedding_update = self.learning_rate * (sigmoid * (pos_item_embedding - neg_item_embedding) - self.reg * user_embedding)
        pos_item_embedding_update = self.learning_rate * (sigmoid * user_embedding - self.reg * pos_item_embedding)
        neg_item_embedding_update = self.learning_rate * (-sigmoid * user_embedding - self.reg * neg_item_embedding)

        self.user_embeddings[user] += user_embedding_update
        self.item_embeddings[pos_item] += pos_item_embedding_update
        self.item_embeddings[neg_item] += neg_item_embedding_update

if __name__ == "__main__":
    num_users = 10
    num_items = 15
    embedding_dim = 5
    user_item_pairs = [(0, 1), (1, 2), (2, 3), (3, 4)]

    bpr = BPR(num_users, num_items, embedding_dim)
    bpr.train(user_item_pairs, num_epochs=10)
````

### Explanation of the update logic

1. **Initialization**: When we initialize the BPR class, we randomly generate embeddings (vectors of numbers) for users and items. These embeddings represent the latent features of users and items.

2. **Training**: During training, for each user and their positive item, we randomly sample a negative item. A positive item is one that the user has interacted with, while a negative item is one they haven't.

3. **Update**: The core part of the training is the update step, where we adjust the embeddings of the user, the positive item, and the negative item. Here’s a simplified explanation:
   - **User Embedding**: We adjust the user's embedding to be closer to the positive item and farther from the negative item.
   - **Positive Item Embedding**: We adjust the positive item's embedding to be closer to the user's embedding.
   - **Negative Item Embedding**: We adjust the negative item's embedding to be farther from the user's embedding.

4. **Sigmoid Function**: The sigmoid function is used to calculate the probability that the user prefers the positive item over the negative item. If this probability is high, the updates will be small, and if it's low, the updates will be larger.

5. **Learning Rate and Regularization**: The learning rate controls how big the updates are. Regularization is used to prevent overfitting by penalizing large values in the embeddings.

The goal of these updates is to learn embeddings that can predict which items a user will prefer, by bringing the embeddings of users and their preferred items closer together in the latent space.
