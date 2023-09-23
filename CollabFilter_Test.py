import implicit
import numpy as np
from scipy.sparse import csr_matrix
from datetime import datetime, timedelta
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


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
    {"videoId": 3, "userId": 0, "timeWatched": 0.7},
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


def decompose_data(data: list[dict]):
    # Include all possible video Ids in the encoder
    video_ids = np.array([x['videoId'] for x in data])
    user_ids = np.array([x['userId'] for x in data])

    # Process timeWatched
    affiliation = np.array([x['timeWatched'] for x in data])

    return video_ids, user_ids, affiliation


def display_points(points, color, transform: None):
    pca = PCA(n_components=2)
    points_2d = pca.fit_transform(points)

    plt.scatter(points_2d[:,0], points_2d[:,1])



if __name__ == "__main__":
    video_ids, user_ids, affiliation = decompose_data(interactions)

    n_videos = video_ids.max() + 1
    n_users = user_ids.max() + 1

    # Create a CSR Matrix
    interaction_matrix = csr_matrix((affiliation, (user_ids, video_ids)), shape=(n_users, n_videos))
    # print(interaction_matrix)

    # Fit the model
    model = implicit.als.AlternatingLeastSquares(factors=20, random_state=0)
    model.fit(interaction_matrix)

    user_factors = model.user_factors
    item_factors = model.item_factors

    user_pca = PCA(n_components=2)
    # combined = np.concatenate((user_factors, item_factors))
    user_factors_2d = user_pca.fit_transform(user_factors)

    item_pca = PCA(n_components=2)
    item_factors_2d = item_pca.fit_transform(item_factors)

    # plt.scatter(user_factors_2d[:,0], user_factors_2d[:,1], c='blue')
    # plt.show()

    # print(user_factors_2d.shape)
    # print(user_factors_2d)

    video_ids, user_ids, affiliation = decompose_data(new_interactions)

    interaction_matrix = interaction_matrix.tolil()
    interaction_matrix[user_ids, video_ids] = affiliation

    interaction_matrix = interaction_matrix.tocsr()

    # print(user_ids)
    # print(interaction_matrix)
    # model.partial_fit_users(user_ids, interaction_matrix[user_ids])
    model.fit(interaction_matrix)

    user_factors = model.user_factors
    item_factors = model.item_factors
    
    new_user_factors_2d = user_pca.transform(user_factors)
    new_item_factors_2d = item_pca.transform(item_factors)

    # print(new_user_factors_2d.shape)
    # print(new_user_factors_2d)


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
    for i in range(n_videos):
        plt.text(item_factors_2d[i,0], item_factors_2d[i,1], i)

    plt.scatter(new_item_factors_2d[:,0], new_item_factors_2d[:,1], c='red', marker='x')
    for i in range(n_videos):
        plt.text(new_item_factors_2d[i,0], new_item_factors_2d[i,1], i)

    plt.show()


    # print(user_factors.shape)
    # print(user_factors)

#     try:
#         print(f"Target user: {model.user_factors.shape}")
#         encoded_target_user = user_enc.transform([target_user])[0]
#         target_user_embedding = model.user_factors[encoded_target_user]
#     except ValueError:
#         print(f"User {target_user} does not exist in the data.")
#         return None
    
#     # Videos that the target user has interacted with in the last 3 days
#     three_days_ago = datetime.now() - timedelta(days=3)
#     recently_seen = {x['videoId'] for x in data if x['userId'] == target_user and datetime.fromtimestamp(x['dateWatched'] / 1000) > three_days_ago}

#     # Remove recently seen videos from all_media_ids
#     filtered_media_ids = [video for video in all_media_ids if video not in recently_seen]
#     encoded_filtered_media_ids = video_enc.transform(filtered_media_ids).astype(int)

#     # Calculate recommendation scores for these videos
#     scores = model.item_factors.dot(model.user_factors[encoded_target_user].T)

#     # Filter only those indices that are within the bounds of the `scores` array
#     filtered_indices = encoded_filtered_media_ids[encoded_filtered_media_ids < len(scores)]

#     # Get the scores corresponding to the filtered indices
#     filtered_scores = scores[filtered_indices]

#     # Get top 20 recommendations
#     top_indices = np.argsort(-filtered_scores)[:20]
#     top_recommendations = filtered_indices[top_indices]

#     # Decode to original Firestore document IDs
#     original_video_ids = video_enc.inverse_transform(top_recommendations)
#     print(f"embedding for {target_user_id}: {target_user_embedding}")
#     return original_video_ids

# # Example usage
# target_user_id = 'gCJT7Qd3SAclfHHJYf3Mc42QFh33'
# recommended_videos = get_recommendations(target_user_id)
# if get_recommendations(target_user_id) is not None:
#     print(f"{recommended_videos}")
# else:
#     print(f"No recommendations available for {target_user_id}.")










