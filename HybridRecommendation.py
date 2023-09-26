import implicit
import openai
from scipy.sparse import csr_matrix
import numpy as np
from numpy.random import RandomState, SeedSequence, MT19937


test_item_data = [
    {
        'title': "Exploring the Mysteries of Deep Space",
        'description': "Embark on a journey through the cosmos and discover the wonders of distant galaxies."
    },
    {
        'title': "Cooking Masterclass: Italian Cuisine",
        'description': "Learn the art of making mouthwatering Italian dishes from a renowned chef."
    },
    {
        'title': "Epic Adventure: Lost City of Atlantis",
        'description': "Join a daring expedition to uncover the secrets of the legendary lost city beneath the ocean's depths."
    },
    {
        'title': "Mindfulness Meditation for Stress Relief",
        'description': "Find inner peace and reduce stress with guided mindfulness meditation sessions."
    },
    {
        'title': "Incredible Wildlife of the Amazon Rainforest",
        'description': "Witness the astonishing diversity of creatures that call the Amazon rainforest home."
    },
    {
        'title': "Virtual Reality Gaming Revolution",
        'description': "Experience the future of gaming with cutting-edge virtual reality technology."
    },
    {
        'title': "Art of Origami: Paper Folding Mastery",
        'description': "Learn the ancient art of origami and create intricate paper sculptures."
    },
    {
        'title': "The Science of Time Travel",
        'description': "Explore the theoretical possibilities and paradoxes of traveling through time."
    },
    {
        'title': "Culinary Adventure: Street Food Around the World",
        'description': "Savor the flavors of global street food markets in this culinary journey."
    },
    {
        'title': "Secrets of the Egyptian Pyramids",
        'description': "Uncover the enigmatic history and construction techniques behind the pyramids of Egypt."
    },
    {
        'title': "Robotics and Artificial Intelligence Revolution",
        'description': "Discover how AI and robots are shaping the future of industry and daily life."
    },
    {
        'title': "Oceanic Wonders: Coral Reefs and Marine Life",
        'description': "Dive into the vibrant world of coral reefs and the creatures that inhabit them."
    },
    {
        'title': "The Art of Movie Making",
        'description': "Go behind the scenes to learn the secrets of creating blockbuster films."
    },
    {
        'title': "Yoga for Mind, Body, and Soul",
        'description': "Find balance and tranquility through yoga practices that nurture your well-being."
    },
    {
        'title': "Historical Mysteries: Unsolved Ancient Riddles",
        'description': "Delve into the unsolved mysteries of ancient civilizations and lost cultures."
    },
    {
        'title': "SpaceX: Journey to Mars",
        'description': "Follow SpaceX's ambitious mission to send humans to the red planet."
    },
    {
        'title': "The World of Underwater Photography",
        'description': "Capture the beauty of the underwater world through the lens of a camera."
    },
    {
        'title': "Virtual Art Gallery Tour",
        'description': "Explore renowned art collections from around the world without leaving your home."
    },
    {
        'title': "The Psychology of Dreams",
        'description': "Unravel the mysteries of the human mind by delving into the realm of dreams."
    },
    {
        'title': "Eco-Friendly Living: Sustainable Practices",
        'description': "Learn how to adopt eco-friendly habits and reduce your environmental footprint."
    },
    {
        'title': "Adventures in Space Tourism",
        'description': "Discover how space tourism is becoming a reality for everyday adventurers."
    },
    {
        'title': "Mystical Legends: Dragons of Mythology",
        'description': "Explore the fascinating world of dragon mythology from different cultures."
    },
    {
        'title': "The Science of Superheroes",
        'description': "Uncover the scientific principles behind the superpowers of your favorite heroes."
    },
    {
        'title': "Exploring Haunted Places: Paranormal Investigations",
        'description': "Join ghost hunters as they explore haunted locations and seek evidence of the supernatural."
    },
    {
        'title': "A Journey through Classical Music",
        'description': "Immerse yourself in the timeless beauty of classical music compositions."
    },
    {
        'title': "Sustainable Farming: Farm-to-Table Revolution",
        'description': "Discover the benefits of sustainable farming practices and the farm-to-table movement."
    },
    {
        'title': "The Lost World of Dinosaurs",
        'description': "Travel back in time to the prehistoric era and encounter the giants that once ruled the Earth."
    },
    {
        'title': "The Art of Mixology: Craft Cocktails",
        'description': "Master the art of crafting delicious cocktails and impress your guests with your bartending skills."
    }
]


def generate_interaction_data(item_data, N=50, n_users=10, seed: int | None = None):
    """Generate random user-item interaction data"""

    np.random.seed(seed) # Set random seed for reproducibility

    # Initialize watch history data
    user_data = [
        {
            'item_ids': np.array([], dtype=np.int32),
            'affiliation': np.array([])
        } for _ in range(n_users)
    ]

    for i in range(len(item_data)):
        item_data[i]['user_ids'] = np.array([], dtype=np.int32),
        item_data[i]['affiliation'] = np.array([])
        item_data[i]['views'] = 0

    # Generate random interactions
    user_ids = np.random.choice(n_users, size=N, replace=True)
    item_ids = np.random.choice(len(item_data), size=N, replace=True)
    affiliation = np.random.rand(N)

    # Add interactions to user and item data
    for i in range(N):
        user_data[user_ids[i]]['item_ids'] = np.append(user_data[user_ids[i]]['item_ids'], item_ids[i])
        user_data[user_ids[i]]['affiliation'] = np.append(user_data[user_ids[i]]['affiliation'], affiliation[i])

        item_data[item_ids[i]]['user_ids'] = np.append(item_data[item_ids[i]]['user_ids'], user_ids[i])
        item_data[item_ids[i]]['affiliation'] = np.append(item_data[item_ids[i]]['affiliation'], affiliation[i])
        item_data[item_ids[i]]['views'] += 1
    
    return user_data, item_data



class CollaborativeFiltering:

    def __init__(self):
        self.model = implicit.als.AlternatingLeastSquares(factors=50, regularization=0.01, iterations=50)
        self.user_data = []
    
    def _csr_from_user_data(self, user_data, shape=None):
        # Generate CSR matrix from user data
        cols = np.array([], dtype=np.int32)
        rows = np.array([], dtype=np.int32)
        affiliation = np.array([])

        for u in range(len(user_data)):
            item_ids = user_data[u]["item_ids"]
            cols = np.append(cols, item_ids, axis=0)
            rows = np.append(rows, u * np.ones_like(item_ids, dtype=np.int32), axis=0)
            affiliation = np.append(affiliation, user_data[u]["affiliation"], axis=0)
        
        return csr_matrix((affiliation, (rows, cols)), shape=shape)
    
    def fit_from_user_data(self, user_data, shape=None):
        matrix = self._csr_from_user_data(user_data, shape=shape)

        self.model.fit(matrix, show_progress=False)
    
    def get_recommendations(self, user_ids, user_data, n=5):
        matrix = self._csr_from_user_data(user_data)

        self.model.partial_fit_users(user_ids, matrix)

        return self.model.recommend(user_ids, matrix, n)


class ContentBasedRecommendation:

    def __init__(self):
        self.items = []
        self.embeddings = None

    def _get_embeddings(self, text: str | list[str], model="text-embedding-ada-002"):
        # Clean string(s)
        if isinstance(text, str):
            text = [text]
        text = [subtext.replace("\n", " ") for subtext in text]

        embeddings = openai.Embedding.create(input=text, model=model)['data']

        return np.array([embedding["embedding"] for embedding in embeddings])
    
    def add_items(self, items: str | list[str]):
        if isinstance(items, str):
            items = [items]
        self.items.extend(items)
        if self.embeddings is None:
            self.embeddings = self._get_embeddings(items)
        else:
            self.embeddings = np.append(self.embeddings, self._get_embeddings(items), axis=0)
    
    def query(self, query: str | list[str], n=5):
        if isinstance(query, str):
            query = [query]
        query_embedding = self._get_embeddings(query)
        cos_similarity = query_embedding @ self.embeddings.T

        return np.argsort(cos_similarity)[::-1][:n]
    
    def get_user_embedding(self, user_data, n=5):
        # Compute weights from affiliation scores (softmax)
        a_exp = np.exp(user_data["affiliation"])
        weights = a_exp / np.sum(a_exp)

        # Compute weighted average
        weighted_avg = np.average(self.embeddings[user_data["item_ids"]], axis=0, weights=weights)

        return weighted_avg
    
    def get_recommendations(self, user_data, n=5):
        user_embedding = self.get_user_embedding(user_data)

        # Remove items that the user has already seen
        mask = np.ones(len(self.items), dtype=bool)
        mask[user_data["item_ids"]] = False

        candidate_embeddings = self.embeddings[mask]
        offsets = np.cumsum(~mask)[mask]

        # Compute cosine similarity
        cos_similarity = user_embedding @ candidate_embeddings.T

        rec_idx = np.argsort(cos_similarity)[::-1][:n]

        return rec_idx + offsets[rec_idx], cos_similarity[rec_idx]
    


class HybridRecommendation:

    def __init__(self):
        self.cf = CollaborativeFiltering()
        self.cb = ContentBasedRecommendation()
    
    def add_data(self, user_data, item_data):
        self.cf.fit_from_user_data(user_data)
        self.cb.add_items([f"{item['title']} - {item['description']}" for item in item_data])
    
    def get_recommendations(self, user_id, user_data, item_data, n=5):
        cf_recs, cf_scores = self.cf.get_recommendations([user_id], [user_data], n=n//2)
        cb_recs, cb_scores = self.cb.get_recommendations(user_data, n=n-(n//2))

        combined_recs = np.append(cf_recs, cb_recs)

        # Sort recommendations by views
        item_views = np.array([item['views'] for item in item_data])
        sorted_idx = np.argsort(item_views[combined_recs])[::-1]
        combined_recs = combined_recs[sorted_idx]

        return combined_recs, item_views[combined_recs]
    


if __name__ == "__main__":
    test_user_data, test_item_data = generate_interaction_data(test_item_data, N=500, n_users=20, seed=42)

    hybrid = HybridRecommendation()
    hybrid.add_data(test_user_data, test_item_data)
    recommendations = hybrid.get_recommendations(0, test_user_data[0], test_item_data, n=10)
    print(recommendations)
