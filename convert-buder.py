
import numpy as np
import pickle
import thecannon as tc
import matplotlib.pyplot as plt

np.random.seed(0)


print("Loading data")
with open("Cannon3.4_Sp_SMEmasks_model.pickle", "rb") as fp:
    spectra, training_set_labels, label_names, offsets, coeffs, covs, scatters, chis, chisqs \
        = pickle.load(fp, encoding="latin-1")

# Save memory by deleting things we won't use.
del covs, chis, chisqs

dispersion = np.hstack([
    np.arange(4715.94, 4896.00, 0.046), # ab lines 4716.3 - 4892.3
    np.arange(5650.06, 5868.25, 0.055), # ab lines 5646.0 - 5867.8
    np.arange(6480.52, 6733.92, 0.064), # ab lines 6481.6 - 6733.4
    np.arange(7693.50, 7875.55, 0.074), # ab lines 7691.2 - 7838.5

])

# Create a cannon model.
print("Constructing model")
vectorizer = tc.vectorizer.PolynomialVectorizer(label_names.astype('str'), 2)

training_set_flux, training_set_err = spectra.T
training_set_ivar = 1.0/training_set_err**2

model = tc.CannonModel(training_set_labels, training_set_flux, training_set_ivar,
                       vectorizer, dispersion=dispersion)
model._s2 = scatters**2
model._theta = coeffs
model._fiducials = offsets
model._scales = np.ones_like(offsets)

# Check a random spectrum.
idx = np.random.choice(len(training_set_flux))

fig, ax = plt.subplots()
ax.plot(dispersion, training_set_flux[idx], c='k')

model_flux = model(training_set_labels[idx]).flatten()
ax.plot(dispersion, model_flux, c='r')

initial_labels = np.percentile(training_set_labels, [5, 50, 95], axis=0)

test_labels, test_cov, meta = model.test(training_set_flux, training_set_ivar,
                                         initial_labels=initial_labels)

for i, label_name in enumerate(label_names):

    fig, ax = plt.subplots()
    ax.scatter(training_set_labels.T[i], test_labels.T[i])
    limits = np.hstack([ax.get_xlim(), ax.get_ylim()])
    limits = (np.min(limits), np.max(limits))
    ax.plot(limits, limits, c="#666666", linestyle=":", zorder=-1)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    diff = test_labels.T[i] - training_set_labels.T[i]
    mean, sigma = np.mean(diff), np.std(diff)
    ax.set_title(f"{label_name.decode()}: {mean:.2f} +/- {sigma:.2f}")

    ax.set_xlabel("training set label")
    ax.set_ylabel("test set label")

    fig.tight_layout()

print("Writing model")
model.write("galah.model", include_training_set_spectra=True, overwrite=True)


