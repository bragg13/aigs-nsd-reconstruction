# %% open all vectors
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from orbax.checkpoint.standard_checkpointer import StandardSave
import scipy.stats as stats
from pingouin import multivariate_normality
PROJECT_DIR = '/Users/andrea/Desktop/aigs/simple_autoencoder/' # repetition
subjects = [1, 2, 3, 4, 5, 6, 7, 8]

# load the latent vectors for each participant
latent_vectors = { f'subj{index}':{'person':None, 'not_person':None} for index in subjects } # structure to save latent vectors
norms = { f'subj{index}':{'person':None, 'not_person':None} for index in subjects } # structure to save norms

for i in range(1, 9):
    print(f'loading subject {i}')

    # this is a list of latent vectors
    person = jnp.load(f'{PROJECT_DIR}results/shared/subj0{i}_shared_person.npy')
    not_person = jnp.load(f'{PROJECT_DIR}results/shared/subj0{i}_shared_not_person.npy')

    # check if the arrays are equal by mistake
    # print(jnp.all(person == not_person))
    print(person.shape)

    # save latent vectors
    latent_vectors[f'subj{i}']['person'] = person # matrix of shape (382, 128)
    latent_vectors[f'subj{i}']['not_person'] = not_person

    # check normality
    # if pval > 0.05, the data is (multivariately?) normally distributed
    print(multivariate_normality(person, alpha=0.05))
    print(multivariate_normality(not_person, alpha=0.05))

    # save norms
    norm_person = np.linalg.norm(person, axis=1) # axis 1 because I want to get the norm of each lv
    norm_not_person = np.linalg.norm(not_person, axis=1)
    norms[f'subj{i}']['person'] = norm_person
    norms[f'subj{i}']['not_person'] = norm_not_person
# %% check if they are normally distributed
fig, axs = plt.subplots(8, 2, figsize=(12, 15))
for i in range(1, 9):
    stats.probplot(norms[f'subj{i}']['person'], dist="norm", plot=axs[i-1, 0])
    stats.probplot(norms[f'subj{i}']['not_person'], dist="norm", plot=axs[i-1, 1])
    axs[i-1, 0].set_title(f'subj{i} person')
    axs[i-1, 1].set_title(f'subj{i} not person')
    axs[i-1, 0].grid(True)
    axs[i-1, 1].grid(True)

plt.tight_layout()
plt.show()

# %% just to visualise the latent vectors
import numpy as np
fig, axs = plt.subplots(4, 2, figsize=(12, 15))
i = 0
j = 0
for index, subject in enumerate(subjects):
    person = latent_vectors[f'subj{subject}']['person']
    not_person = latent_vectors[f'subj{subject}']['not_person']
    person_non_zero = person[index][person[index] != 0]
    not_person_non_zero = not_person[index][not_person[index] != 0]
    axs[i, j].scatter(np.arange(0, len(person_non_zero), 1), person_non_zero, color='red', alpha=0.6)
    axs[i, j].scatter(np.arange(0, len(not_person_non_zero), 1), not_person_non_zero, color='blue', alpha=0.6)
    axs[i, j].set_title(f'subj{subject}')
    i += 1
    if i == 4:
        i = 0
        j = 1
    plt.suptitle("Latent Vectors of Person and Non-Person Images")

# %%
# perform a u-test to check if the norms are significantly different
# im comparing observations of different images from the same subject, so I need to use an indipendent t-test
# Person - Non-person
import numpy as np
for subject in subjects:
    person = norms[f'subj{subject}']['person']
    not_person = norms[f'subj{subject}']['not_person']

    print(stats.mannwhitneyu(person, not_person))
#
#
fig, axs = plt.subplots(4, 2, figsize=(12, 15))
for i in range(1, 9):
    axs[(i-1)//2, (i-1)%2].scatter(range(len(norms[f'subj{i}']['person'])), norms[f'subj{i}']['person'], color='blue', label='person', alpha=0.6)
    axs[(i-1)//2, (i-1)%2].scatter(range(len(norms[f'subj{i}']['not_person'])), norms[f'subj{i}']['not_person'], color='red', label='not person', alpha=0.6)
    axs[(i-1)//2, (i-1)%2].set_title(f'subj{i}')
    axs[(i-1)//2, (i-1)%2].legend()
    axs[(i-1)//2, (i-1)%2].grid(True)

plt.tight_layout()
plt.show()


# %%
# lets now use PCA instead
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

fig, axs = plt.subplots(4, 2, figsize=(12, 15))
pca = PCA(n_components=2)
i = 0
j = 0
for subj in subjects:
    person_vecs = latent_vectors[f'subj{subj}']['person']
    not_person_vecs = latent_vectors[f'subj{subj}']['not_person']
    p = pca.fit_transform(StandardScaler().fit_transform(person_vecs))
    np = pca.fit_transform(StandardScaler().fit_transform(not_person_vecs))
    axs[i, j].scatter(p[:, 0], p[:, 1], color='red', alpha=0.7)
    axs[i, j].scatter(np[:, 0], np[:, 1], color='blue', alpha=0.7)
    axs[i, j].grid(True)
    axs[i, j].set_title(f'subj{subj}')
    i += 1
    if i == 4:
        i = 0
        j = 1


plt.suptitle("2D PCA Representation of Data")
plt.show()
