import implicit
import numpy as np
from scipy.sparse import csr_matrix
from datetime import datetime, timedelta
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


n_users = 5
n_items = 10

interactions = [
    {"videoId": 1, "userId": 1, "timeWatched": 0.5},
    {"videoId": 2, "userId": 1, "timeWatched": 0.8},
    {"videoId": 3, "userId": 1, "timeWatched": 0.3},
    {"videoId": 4, "userId": 2, "timeWatched": 0.6},
    {"videoId": 5, "userId": 2, "timeWatched": 1.0},
    {"videoId": 6, "userId": 2, "timeWatched": 0.7},
    {"videoId": 7, "userId": 3, "timeWatched": 0.9},
    {"videoId": 8, "userId": 3, "timeWatched": 0.4},
    {"videoId": 9, "userId": 3, "timeWatched": 0.2},
    {"videoId": 0, "userId": 4, "timeWatched": 0.85},
    {"videoId": 1, "userId": 4, "timeWatched": 0.5},
    {"videoId": 2, "userId": 4, "timeWatched": 0.95},
    {"videoId": 4, "userId": 0, "timeWatched": 0.3},
    {"videoId": 5, "userId": 0, "timeWatched": 0.8},
    {"videoId": 6, "userId": 1, "timeWatched": 0.6},
    {"videoId": 7, "userId": 1, "timeWatched": 0.2},
    {"videoId": 8, "userId": 2, "timeWatched": 0.75},
    {"videoId": 9, "userId": 2, "timeWatched": 0.95},
    {"videoId": 0, "userId": 3, "timeWatched": 0.55},
    {"videoId": 1, "userId": 3, "timeWatched": 0.7},
    {"videoId": 2, "userId": 0, "timeWatched": 0.6},
    {"videoId": 3, "userId": 0, "timeWatched": 0.9},
    {"videoId": 4, "userId": 4, "timeWatched": 0.65},
    {"videoId": 5, "userId": 4, "timeWatched": 0.8},
    {"videoId": 6, "userId": 0, "timeWatched": 0.4},
    {"videoId": 7, "userId": 2, "timeWatched": 0.6},
    {"videoId": 8, "userId": 1, "timeWatched": 0.9},
    {"videoId": 9, "userId": 4, "timeWatched": 0.1},
    {"videoId": 0, "userId": 0, "timeWatched": 0.85}
]

new_interactions = [
    {"videoId": 1, "userId": 1, "timeWatched": 1},
    {"videoId": 1, "userId": 2, "timeWatched": 1},
    {"videoId": 1, "userId": 3, "timeWatched": 1},
]

# user_data = [
#     {
#         'item_ids': np.array([4, 5, 2, 3, 6, 0], dtype=np.int32),
#         'affiliation': np.array([0.3, 0.8, 0.6, 0.9, 0.4, 0.85])
#     }, 
#     {
#         'item_ids': np.array([1, 2, 3, 6, 7, 8], dtype=np.int32), 
#         'affiliation': np.array([0.5, 0.8, 0.3, 0.6, 0.2, 0.9])
#     }, 
#     {
#         'item_ids': np.array([4, 5, 6, 8, 9, 7], dtype=np.int32),
#         'affiliation': np.array([0.6, 1., 0.7, 0.75, 0.95, 0.6])
#     }, 
#     {
#         'item_ids': np.array([7, 8, 9, 0, 1], dtype=np.int32), 
#         'affiliation': np.array([0.9, 0.4, 0.2, 0.55, 0.7])
#     }, 
#     {
#         'item_ids': np.array([0, 1, 2, 4, 5, 9], dtype=np.int32), 
#         'affiliation': np.array([0.85, 0.5, 0.95, 0.65, 0.8, 0.1])
#     }
#     ]


def decompose_data(data: list[dict]):
    # Include all possible video Ids in the encoder
    video_ids = np.array([x['videoId'] for x in data], dtype=np.int32)
    user_ids = np.array([x['userId'] for x in data], dtype=np.int32)

    # Process timeWatched
    affiliation = np.array([x['timeWatched'] for x in data])


    return video_ids, user_ids, affiliation


def interactions_to_user_data(interaction_data: list[dict]) -> list[np.ndarray]:
    item_ids, user_ids, affiliation = decompose_data(interaction_data)

    user_data = [{ 
        "item_ids": np.array([], dtype=np.int32),
        "affiliation": np.array([])
    } for _ in range(n_users)]

    for i in range(len(user_ids)):
        user_data[user_ids[i]]["item_ids"] = np.append(user_data[user_ids[i]]["item_ids"], np.array([item_ids[i]]))
        user_data[user_ids[i]]["affiliation"] = np.append(user_data[user_ids[i]]["affiliation"], np.array([affiliation[i]]))
    
    return user_data


def interactions_to_item_data(interaction_data: list[dict]) -> list[np.ndarray]:
    item_ids, user_ids, affiliation = decompose_data(interaction_data)

    item_data = [{ 
        "user_ids": np.array([], dtype=np.int32),
        "affiliation": np.array([])
    } for _ in range(n_items)]

    for i in range(len(item_ids)):
        item_data[item_ids[i]]["user_ids"] = np.append(item_data[item_ids[i]]["user_ids"], np.array([user_ids[i]]))
        item_data[item_ids[i]]["affiliation"] = np.append(item_data[item_ids[i]]["affiliation"], np.array([affiliation[i]]))
    
    return item_data


def update_data(user_data: list[dict], item_data: list[dict], new_interactions: list[dict]):
    item_ids, user_ids, affiliation = decompose_data(new_interactions)
    
    for i in range(len(new_interactions)):
        # Update user data
        if item_ids[i] in user_data[user_ids[i]]["item_ids"]:
            index = np.where(user_data[user_ids[i]]["item_ids"] == item_ids[i])
            user_data[user_ids[i]]["affiliation"][index] = affiliation[i]
        else:
            user_data[user_ids[i]]["item_ids"] = np.append(user_data[user_ids[i]]["item_ids"], np.array([item_ids[i]]))
            user_data[user_ids[i]]["affiliation"] = np.append(user_data[user_ids[i]]["affiliation"], np.array([affiliation[i]]))
        
        # Update item data
        if user_ids[i] in item_data[item_ids[i]]["user_ids"]:
            index = np.where(item_data[item_ids[i]]["user_ids"] == user_ids[i])
            item_data[item_ids[i]]["affiliation"][index] = affiliation[i]
        else:
            item_data[item_ids[i]]["user_ids"] = np.append(item_data[item_ids[i]]["user_ids"], np.array([user_ids[i]]))
            item_data[item_ids[i]]["affiliation"] = np.append(item_data[item_ids[i]]["affiliation"], np.array([affiliation[i]]))
            

def get_model(full_interactions):
    # Reformat interaction data
    item_ids, user_ids, affiliation = decompose_data(full_interactions)

    # Create a CSR Matrix
    interaction_matrix = csr_matrix((affiliation, (user_ids, item_ids)), shape=(n_users, n_items))
    print(interaction_matrix.todense())

    # Fit the model
    model = implicit.als.AlternatingLeastSquares(factors=20, random_state=0)
    model.fit(interaction_matrix)

    return model


def fit_model_from_user_data(user_data):
    # Generate CSR matrix from user data
    cols = np.array([])
    rows = np.array([])
    affiliation = np.array([])

    for u in range(n_users):
        user_idx = u
        item_ids = user_data[user_idx]["item_ids"]

        cols = np.append(cols, item_ids, axis=0)
        rows = np.append(rows, u * np.ones_like(item_ids), axis=0)
        affiliation = np.append(affiliation, user_data[user_idx]["affiliation"], axis=0)
    
    matrix = csr_matrix((affiliation, (rows, cols)), shape=(n_users, n_items))
    print(matrix.todense())

    model.fit(matrix)

    return model


def update_user_factors(user_ids: np.ndarray, user_data: list[dict], model):
    # Generate CSR matrix from user data
    cols = np.array([])
    rows = np.array([])
    affiliation = np.array([])

    for u in range(len(user_ids)):
        user_idx = user_ids[u]
        item_ids = user_data[user_idx]["item_ids"]

        cols = np.append(cols, item_ids, axis=0)
        rows = np.append(rows, u * np.ones_like(item_ids), axis=0)
        affiliation = np.append(affiliation, user_data[user_idx]["affiliation"], axis=0)
    
    matrix = csr_matrix((affiliation, (rows, cols)), shape=(len(user_ids), n_items))
    print(matrix.todense())

    # Update factors for these users
    model.partial_fit_users(user_ids, matrix)

    return model


def update_item_factors(item_ids: np.ndarray, item_data: list[np.ndarray], model):
    # Generate CSR matrix from user data
    cols = np.array([])
    rows = np.array([])
    affiliation = np.array([])

    for i in range(len(item_ids)):
        item_idx = item_ids[i]
        user_ids = item_data[item_idx]["user_ids"]

        cols = np.append(cols, user_ids, axis=0)
        rows = np.append(rows, i * np.ones_like(user_ids), axis=0)
        affiliation = np.append(affiliation, item_data[item_idx]["affiliation"], axis=0)

    matrix = csr_matrix((affiliation, (rows, cols)), shape=(len(item_ids), n_users))
    print(matrix.T.todense())

    # Update factors for these users
    model.partial_fit_items(item_ids, matrix)

    return model
        
    
def plot_before_after(user_factors_2d, item_factors_2d, new_user_factors, new_item_factors, user_pca, item_pca):
    # Get 2d representation of new user and item factors
    new_user_factors_2d = user_pca.transform(model.user_factors)
    new_item_factors_2d = item_pca.transform(model.item_factors)

    # Display before and after of user and item factors
    plt.subplot(1, 2, 1)
    plt.title("User Factors")
    plt.scatter(user_factors_2d[:,0], user_factors_2d[:,1], c='blue')
    for i in range(n_users):
        plt.text(user_factors_2d[i,0], user_factors_2d[i,1], i)

    plt.scatter(new_user_factors_2d[:,0], new_user_factors_2d[:,1], c='red', marker='x')
    for i in range(n_users):
        plt.text(new_user_factors_2d[i,0], new_user_factors_2d[i,1], i)

    plt.subplot(1, 2, 2)
    plt.title("Item Factors")
    plt.scatter(item_factors_2d[:,0], item_factors_2d[:,1], c='blue')
    for i in range(n_items):
        plt.text(item_factors_2d[i,0], item_factors_2d[i,1], i)

    plt.scatter(new_item_factors_2d[:,0], new_item_factors_2d[:,1], c='red', marker='x')
    for i in range(n_items):
        plt.text(new_item_factors_2d[i,0], new_item_factors_2d[i,1], i)

    plt.show()


if __name__ == "__main__":
    
    # Train the model off of the initial interaction data
    model = get_model(interactions)

    # Use PCA to compute 2d representation of user and item factors
    user_pca = PCA(n_components=2)
    user_factors_2d = user_pca.fit_transform(model.user_factors)

    item_pca = PCA(n_components=2)
    item_factors_2d = item_pca.fit_transform(model.item_factors)

    
    # Update user factors with new data
    user_data = interactions_to_user_data(interactions)
    item_data = interactions_to_item_data(interactions)

    update_data(user_data, item_data, new_interactions)
    _, user_ids, _ = decompose_data(new_interactions)

    # Update user factors
    model = update_user_factors(user_ids, user_data, model)
    plot_before_after(user_factors_2d, item_factors_2d, model.user_factors, model.item_factors, user_pca, item_pca)

    # Update item factors
    model = update_item_factors(np.arange(10), item_data, model)
    plot_before_after(user_factors_2d, item_factors_2d, model.user_factors, model.item_factors, user_pca, item_pca)

    # Update remaining user factors
    model = update_user_factors(np.arange(5), user_data, model)
    plot_before_after(user_factors_2d, item_factors_2d, model.user_factors, model.item_factors, user_pca, item_pca)


    # model = fit_model_from_user_data(user_data)

