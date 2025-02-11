import numpy as np
import cv2
import math

def load_grayscale_image(path):
    """
    Loads an image from `path`, converts to grayscale, normalizes to [0,1].
    Returns a float32 2D numpy array (H,W).
    """
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Cannot load image from {path}.")
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return img_gray.astype(np.float32) / 255.0

def fuzzify_ffcm(x):
    """
    Fermatean FCM (FFCM) fuzzification with constraint mu^3 + nu^3 <= 1.
      mu = x
      nu = (1 - x^3)^(1/3)
      pi = (1 - mu^3 - nu^3)^(1/3)
    """
    mu = x
    nu = (1.0 - mu**3)**(1.0/3.0)
    tmp = max(0.0, 1.0 - mu**3 - nu**3)
    pi = tmp**(1.0/3.0)
    return (mu, nu, pi)

def distance_ff(a, b):
    """
    3D Euclidean distance for Fermatean fuzzy vectors a=(mu_a, nu_a, pi_a)
    and b=(mu_b, nu_b, pi_b).
    """
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def initialize_memberships(n_points, c):
    """
    Randomly initialize membership matrix U of shape (n_points, c),
    each row sums to 1.
    """
    U = np.random.rand(n_points, c)
    U /= U.sum(axis=1, keepdims=True)
    return U

def compute_centers_ffcm(X_fuzzy, U, m):
    """
    Compute cluster centers (Fermatean) from memberships U.
    X_fuzzy: list of 3D fuzzy vectors
    U: membership matrix (n_points x c)
    Returns c centers, each a 3D tuple (mu_c, nu_c, pi_c).
    """
    n_points, c = U.shape
    centers = []
    for j in range(c):
        numerator = np.zeros(3, dtype=np.float64)
        denom = 0.0
        for i in range(n_points):
            w = (U[i,j]**m)
            numerator += w * np.array(X_fuzzy[i])
            denom += w
        if denom < 1e-12:
            centers.append((0.0, 0.0, 0.0))
        else:
            center_vec = numerator / denom
            centers.append(tuple(center_vec))
    return centers

def update_memberships_ffcm(X_fuzzy, centers, U_old, m):
    """
    Update membership matrix (standard fuzzy C-means ratio form),
    in 3D Fermatean fuzzy space.
    """
    n_points, c = U_old.shape
    U_new = np.zeros_like(U_old)
    for i in range(n_points):
        dist = [distance_ff(X_fuzzy[i], centers[j]) for j in range(c)]
        for j in range(c):
            denom = 0.0
            for k in range(c):
                if dist[k] < 1e-12:
                    ratio = 1.0 if dist[j] < 1e-12 else 0.0
                else:
                    ratio = (dist[j]/dist[k])**(2.0/(m-1.0))
                denom += ratio
            U_new[i,j] = 1.0 / (denom + 1e-12)
    return U_new

def run_ffcm(X_fuzzy, c=2, m=2.0, max_iter=50, tol=1e-5):
    """
    Runs the iterative FFCM algorithm and returns (U, centers).
    """
    n_points = len(X_fuzzy)
    U = initialize_memberships(n_points, c)
    for _ in range(max_iter):
        centers = compute_centers_ffcm(X_fuzzy, U, m)
        U_new = update_memberships_ffcm(X_fuzzy, centers, U, m)
        diff = np.abs(U_new - U).max()
        U = U_new
        if diff < tol:
            break
    return U, centers

def segment_image(U, H, W):
    """
    Convert membership matrix U (size N x c) into a label image (H x W)
    by picking the cluster with highest membership for each pixel.
    """
    labels = np.argmax(U, axis=1)
    segm = labels.reshape(H, W)
    return segm

# -------------------------------
# EVALUATION METRICS
# -------------------------------

def calc_segmentation_accuracy(seg_mask, gt_mask, c):
    """
    Calculate segmentation accuracy assuming both are label maps in {0,1,...,c-1}.
    Formula:
       SA = ( sum_{i=0..c-1} |A_i ∩ C_i| ) / ( sum_{i=0..c-1} |C_i| )
    Where A_i are pixels labeled i by the segmentation,
          C_i are pixels labeled i by the ground truth.
    """
    total_gt = 0
    intersect_sum = 0
    for i in range(c):
        A_i = (seg_mask == i)
        C_i = (gt_mask == i)
        intersect_sum += np.logical_and(A_i, C_i).sum()
        total_gt     += C_i.sum()
    if total_gt == 0:
        return 0.0
    return intersect_sum / float(total_gt)

def calc_entropy_metrics(img_gray, seg_mask, c):
    """
    Calculate layout entropy (Hl), region entropy (Hr), and total E = Hl + Hr.
    c: number of clusters
    seg_mask: (H,W) label image with values in [0..c-1]
    img_gray: original grayscale image in [0,1], shape (H,W).

    Steps:
      1) For each cluster j, gather all pixels belonging to j => region Rj
      2) S_j = number of pixels in cluster j; S_I = total pixels
      3) layout entropy: Hl(I) = - sum_{j=1..c} (S_j / S_I) * log( S_j / S_I )
      4) region entropy:
         Hr(I) = sum_{j=1..c} ( (S_j / S_I) * H(R_j) )
         where H(R_j) = - sum_{m in V_j} [ L_j(m)/S_j ] * log( L_j(m)/S_j )
         and L_j(m) = number of pixels in region j with gray-level m
         (we can bin intensities or treat them discretely in [0..255]).
    """
    H, W = img_gray.shape
    S_I = H * W
    # layout entropy
    # Count S_j
    s_counts = []
    for j in range(c):
        s_counts.append(np.sum(seg_mask == j))
    s_counts = np.array(s_counts, dtype=np.float64)

    # Hl
    Hl = 0.0
    for j in range(c):
        if s_counts[j] > 0:
            p_j = s_counts[j] / float(S_I)
            Hl += -p_j * math.log(p_j + 1e-12)

    # region entropy Hr
    # We can discretize intensities in [0..255] if originally 0..1 => we do int(255*x)
    # or if your image is float, you can bin them or handle them precisely
    bin_img = (img_gray*255).astype(np.uint8)
    Hr = 0.0
    for j in range(c):
        S_j = s_counts[j]
        if S_j < 1:
            continue
        # gather all pixel intensities in region j
        region_pixels = bin_img[seg_mask == j]
        # count frequency of each m
        freq = np.bincount(region_pixels, minlength=256)  # for 0..255
        # compute H(R_j)
        Hr_j = 0.0
        for m, count in enumerate(freq):
            if count > 0:
                p_m = count / float(S_j)
                Hr_j += -p_m * math.log(p_m + 1e-12)
        # weight by (S_j / S_I)
        Hr += (S_j / float(S_I)) * Hr_j

    E = Hl + Hr
    return Hl, Hr, E

def main():
    # 1) Load an example grayscale image
    path_img = "sample_image.jpg"
    img = load_grayscale_image(path_img)
    H, W = img.shape
    
    # 2) Fuzzify for FFCM
    X_fuzzy = []
    for row in range(H):
        for col in range(W):
            intensity = img[row,col]
            X_fuzzy.append(fuzzify_ffcm(intensity))
    
    # 3) Run FFCM
    c = 3  # number of clusters
    U, centers = run_ffcm(X_fuzzy, c=c, m=2.0, max_iter=50, tol=1e-5)
    
    # 4) Create a segmented image
    segm = segment_image(U, H, W)
    
    # 5) If you have a ground-truth label map, load it for accuracy
    path_gt = "sample_gt.png"
    gt_bgr = cv2.imread(path_gt, cv2.IMREAD_GRAYSCALE)
    gt_mask = gt_bgr  # assuming 0..(c-1) labels
    # or re-map labels if needed
    
    # 6) Calculate segmentation accuracy
    acc = calc_segmentation_accuracy(segm, gt_mask, c)
    print(f"Segmentation Accuracy = {acc:.4f}")
    
    # 7) Calculate entropies
    Hl, Hr, E = calc_entropy_metrics(img, segm, c)
    print(f"Layout Entropy (Hl)     = {Hl:.4f}")
    print(f"Region Entropy (Hr)     = {Hr:.4f}")
    print(f"Total Entropy (E)       = {E:.4f}")

if __name__ == "__main__":
    main()
