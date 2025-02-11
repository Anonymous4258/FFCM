import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
seed = 42
random.seed(seed)
np.random.seed(seed)
def load_grayscale_image(path):
    """
    Load image from `path`, convert to grayscale, normalize to [0,1].
    Returns a float32 2D numpy array (H,W).
    """
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Cannot load image from {path}.")
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_norm = img_gray.astype(np.float32) / 255.0
    return img_norm

# =====================================================================
#  Fuzzification Routines
# =====================================================================

def fuzzify_fcm(x):
    """
    Classical FCM: single dimension representing intensity x in [0,1].
    We'll store it as a tuple of length 1 for uniform handling, e.g. (x,).
    Distance will be 1D absolute difference.
    """
    return (x,)

def fuzzify_ifcm(x):
    """
    Intuitionistic FCM: membership + nonmembership + pi = 1.
    A simple scheme:
      mu = x,
      nu = (1 - x)/2,
      pi = (1 - x)/2.
    Ensures mu + nu + pi = 1.
    """
    mu = x
    nu = (1.0 - x)/2.0
    pi = (1.0 - x)/2.0  # so mu + nu + pi = 1
    return (mu, nu, pi)

def fuzzify_pfcm(x):
    """
    Pythagorean FCM (PFCM) with constraint mu^2 + nu^2 <= 1.
    We'll do:
      mu = x,
      nu = sqrt(max(0, 1 - x^2)),
      pi = sqrt(max(0, 1 - mu^2 - nu^2)).
    """
    mu = x
    nu = np.sqrt(max(0.0, 1.0 - mu**2))
    tmp = max(0.0, 1.0 - mu**2 - nu**2)
    pi = np.sqrt(tmp)
    return (mu, nu, pi)

def fuzzify_ffcm(x):
    """
    Fermatean FCM (FFCM) with constraint mu^3 + nu^3 <= 1.
    We'll do:
      mu = x,
      nu = (1 - x^3)^(1/3),
      pi = (1 - mu^3 - nu^3)^(1/3).
    """
    mu = x
    nu = (1.0 - mu**3)**(1.0/3.0)
    tmp = max(0.0, 1.0 - mu**3 - nu**3)
    pi = tmp**(1.0/3.0)
    return (mu, nu, pi)

# =====================================================================
#  Distance Computation
# =====================================================================

def distance_fuzzy(a, b):
    """
    Computes distance between two fuzzy vectors a and b.
    - If len(a)==1 => treat it as classical FCM in 1D => |a[0] - b[0]|.
    - Else => 3D Euclidean => sqrt((a[0]-b[0])^2 + (a[1]-b[1])^2 + (a[2]-b[2])^2).
    """
    if len(a) == 1:  # classical FCM (1D)
        return abs(a[0] - b[0])
    else:  # IFCM/PFCM/FFCM (3D)
        return np.sqrt( (a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2 )

# =====================================================================
#  FCM Iteration (Generic)
# =====================================================================

def initialize_memberships(num_points, c):
    """
    Randomly init membership matrix U of shape (num_points, c),
    each row sums to 1.
    """
    U = np.random.rand(num_points, c)
    U /= U.sum(axis=1, keepdims=True)
    return U

def compute_centers(X_fuzzy, U, m):
    """
    Compute cluster centers from memberships:
    X_fuzzy: list/array of fuzzy vectors (num_points).
    U: membership (num_points x c).
    Return list of c centers in same dimension as X_fuzzy[0].
    """
    num_points, c = U.shape
    dim = len(X_fuzzy[0])  # 1 or 3
    centers = []
    for j in range(c):
        # Weighted sums
        numerator = np.zeros(dim, dtype=np.float64)
        denom = 0.0
        for i in range(num_points):
            w = (U[i,j]**m)
            # add X_fuzzy[i]*w
            numerator += w * np.array(X_fuzzy[i])
            denom += w
        if denom < 1e-12:
            centers.append(tuple(np.zeros(dim)))
        else:
            center_vec = numerator / denom
            centers.append(tuple(center_vec))
    return centers

def update_memberships(X_fuzzy, centers, U_old, m):
    """
    Update membership matrix via standard FCM ratio formula.
    """
    num_points, c = U_old.shape
    U_new = np.zeros_like(U_old)
    for i in range(num_points):
        dist = [distance_fuzzy(X_fuzzy[i], centers[j]) for j in range(c)]
        for j in range(c):
            # ratio
            denom = 0.0
            for k in range(c):
                # avoid div by zero
                if dist[k] < 1e-12:
                    ratio = 1.0 if dist[j]<1e-12 else 0.0
                else:
                    ratio = (dist[j]/dist[k])**(2.0/(m-1.0))
                denom += ratio
            U_new[i,j] = 1.0 / (denom + 1e-12)
    return U_new

def run_fcm_variant(X_fuzzy, c=2, m=2.0, max_iter=100, tol=1e-5):
    """
    Generic FCM loop:
      X_fuzzy: list of fuzzy vectors
      c: clusters
      m: fuzzifier
    Returns (U, centers).
    """
    num_points = len(X_fuzzy)
    U = initialize_memberships(num_points, c)
    for _ in range(max_iter):
        centers = compute_centers(X_fuzzy, U, m)
        U_new = update_memberships(X_fuzzy, centers, U, m)
        diff = np.abs(U_new - U).max()
        U = U_new
        if diff < tol:
            break
    return U, centers

def segment_image(U, H, W):
    """
    Convert membership matrix U (N x c) into a label image (H x W)
    by taking argmax membership for each pixel.
    """
    labels = np.argmax(U, axis=1)
    segm = labels.reshape(H, W)
    return segm

# =====================================================================
#  MAIN DEMO
# =====================================================================

def main():
    # 1) Load grayscale image
    path = "84.jpg"
    img = load_grayscale_image(path)
    H, W = img.shape
    N = H * W

    # 2) Fuzzify in four ways: FCM(1D), IFCM(3D), PFCM(3D), FFCM(3D).

    # (A) FCM (classical)
    X_fuzzy_fcm = []
    for row in range(H):
        for col in range(W):
            x = img[row, col]  # in [0,1]
            X_fuzzy_fcm.append(fuzzify_fcm(x))

    # (B) IFCM
    X_fuzzy_ifcm = []
    for row in range(H):
        for col in range(W):
            x = img[row, col]
            X_fuzzy_ifcm.append(fuzzify_ifcm(x))

    # (C) PFCM
    X_fuzzy_pfcm = []
    for row in range(H):
        for col in range(W):
            x = img[row, col]
            X_fuzzy_pfcm.append(fuzzify_pfcm(x))

    # (D) FFCM
    X_fuzzy_ffcm = []
    for row in range(H):
        for col in range(W):
            x = img[row, col]
            X_fuzzy_ffcm.append(fuzzify_ffcm(x))

    # 3) Run all four variants
    c = 2          # number of clusters
    m = 2.0        # fuzzifier exponent
    max_iter = 40
    tol = 1e-5

    U_fcm, ctr_fcm     = run_fcm_variant(X_fuzzy_fcm,  c, m, max_iter, tol)
    U_ifcm, ctr_ifcm   = run_fcm_variant(X_fuzzy_ifcm, c, m, max_iter, tol)
    U_pfcm, ctr_pfcm   = run_fcm_variant(X_fuzzy_pfcm, c, m, max_iter, tol)
    U_ffcm, ctr_ffcm   = run_fcm_variant(X_fuzzy_ffcm, c, m, max_iter, tol)

    # 4) Convert memberships to segmentations
    segm_fcm  = segment_image(U_fcm,  H, W)
    segm_ifcm = segment_image(U_ifcm, H, W)
    segm_pfcm = segment_image(U_pfcm, H, W)
    segm_ffcm = segment_image(U_ffcm, H, W)

    # 5) Display comparisons
    fig, axes = plt.subplots(1,5, figsize=(15,4))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("Original")
    axes[0].axis('off')

    axes[1].imshow(segm_fcm, cmap='gray')
    axes[1].set_title("FCM")
    axes[1].axis('off')

    axes[2].imshow(segm_ifcm, cmap='gray')
    axes[2].set_title("IFCM")
    axes[2].axis('off')

    axes[3].imshow(segm_pfcm, cmap='gray')
    axes[3].set_title("PFCM")
    axes[3].axis('off')

    axes[4].imshow(segm_ffcm, cmap='gray')
    axes[4].set_title("FFCM")
    axes[4].axis('off')

    plt.tight_layout()
    plt.savefig("segmentation_comparison.pdf", format="pdf", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
