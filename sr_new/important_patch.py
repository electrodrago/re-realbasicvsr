import cv2
import numpy as np
import json


def count_most_frequent_pixel(image, threshold):
    # Compute the histogram of pixel intensities
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])

    # Find the pixel value with the highest frequency
    most_frequent_pixel = np.argmax(hist)
    frequency = int(hist[most_frequent_pixel])

    if frequency > threshold:
        return most_frequent_pixel
    else:
        return 0


# Divide img into patches
def divide_into_patches(img, division=6):
    # Determine the size of each patch
    patch_size = min(img.shape[0], img.shape[1]) // division

    # Calculate the number of patches in each dimension
    num_patches_x = int(np.ceil(img.shape[1] / patch_size))
    num_patches_y = int(np.ceil(img.shape[0] / patch_size))

    pad = ((0, num_patches_y * patch_size - img.shape[0]), (0, num_patches_x * patch_size - img.shape[1]), (0, 0))

    # Divide the padded img into patches
    patches = []
    for j in range(num_patches_y):
        patches_x = []
        for i in range(num_patches_x):
            x_start = i * patch_size
            y_start = j * patch_size
            patches_x.append((y_start, x_start))
        patches.append(patches_x)

    return patches, patch_size, pad


# Check if the patch contains instance texture
def check_segmentation_region(patch, seg_patch, value, threshold=None):
    if threshold is None:
        # Take threshold to be 40% number of pixels in patch
        threshold = int(patch.shape[0]**2 * 0.4)
    else:
        threshold = int(patch.shape[0]**2 * threshold)

    # Apply the binary mask to the original patch
    binary_mask = np.uint8(seg_patch == value)
    masked_img = cv2.bitwise_and(patch, patch, mask=binary_mask)

    # Count the number of non-zero pixels in the segmentation region
    num_pixels = np.count_nonzero(masked_img)

    # Check if the number of pixels is larger than the threshold
    if num_pixels > threshold:
        return True
    else:
        return False


# Check if the patch is in the region of instance mask
def in_region(patch, patch_size, ins_region):
    # Patch is top, left position
    if ins_region[0] <= patch[0] + patch_size <= ins_region[2] and \
    ins_region[1] <= patch[1] + patch_size <= ins_region[3]:
            return True
    return False


# Input if dict of instance masks
def filter_large_regions(json_dict, filter_size):
    region_filtered = dict()
    for key in json_dict.keys():
        if 'background' in key:
            continue

        # Interest in only the region have large width and height
        box = json_dict[key]["box"]
        if box[2] - box[0] > filter_size * 2 or box[3] - box[1] > filter_size * 2:
            region_filtered[key] = json_dict[key]
            region_filtered[key]['patches'] = []
    # NOTE: region_filtered doesn't contain background
    return region_filtered


# Get region_filtered dict and image_recon details
def get_dict_and_img_recon(img_patches, json_dict, img, mask, filter_size, threshold=None):
    # Obtain the region_filtered dictionary
    region_filtered = filter_large_regions(json_dict, filter_size)
    print(region_filtered.keys())

    # Create a checklist of patches
    visited = [[False] * len(img_patches[0]) for _ in range(len(img_patches))]

    # Create a reconstruction guide from dict patches
    img_recon = [[['background', -1]] * len(img_patches[0]) for _ in range(len(img_patches))]

    # Add ref patches to the instance dict and img_recon, check visited list
    for row in range(len(img_patches)):
        for col in range(len(img_patches[row])):
            for region in region_filtered.keys():
                # patch_properties: tuple contain top left position
                # box: the region of instance segmentation mask
                # print("Check region: ", region)
                patch_properties = img_patches[row][col]
                box = region_filtered[region]['box']

                # Check if the patch is visited or not (added to the instance dict) and in the region
                if visited[row][col] is False and in_region(patch_properties, patch_size, box):
                    top, left = patch_properties[0], patch_properties[1]
                    value = region_filtered[region]['value']

                    # Work on padded img and mask
                    img_patch = img[top:top+patch_size, left:left+patch_size]
                    seg_patch = mask[top:top+patch_size, left:left+patch_size]

                    # cv2.imshow("a", img_patch)
                    # cv2.imshow("b", seg_patch * 40)
                    # cv2.waitKey(0)

                    # Check if the image contains instance texture
                    if check_segmentation_region(img_patch, seg_patch, value, threshold):
                        # If contain, visited set to True, add patch to filter regions, update img_recon
                        region_filtered[region]['patches'].append(img_patch)
                        img_recon[row][col] = [region, len(region_filtered[region]['patches']) - 1]
                        visited[row][col] = True
                
                # if visited[row][col] is True and in_region(patch_properties, patch_size, box):
                #     print("--------------------")
                #     print("Patch in regions: ", region)

    # Add background and non-ref patches, call 'background' together
    region_filtered['background'] = []

    # Add the non-ref patches to background and img_recon
    for row in range(len(img_patches)):
        for col in range(len(img_patches[row])):
            if visited[row][col] is False:
                patch_properties = img_patches[row][col]
                top, left = patch_properties[0], patch_properties[1]
                img_patch = img[top:top+patch_size, left:left+patch_size]

                region_filtered['background'].append(img_patch)
                img_recon[row][col] = ['background', len(region_filtered['background']) - 1]
                visited[row][col] = True
    
    return region_filtered, img_recon


# Get region_filtered dict and image_recon details
def patch_partition(img_patches, json_dict, img, mask, filter_size, threshold=0):
    # Obtain the region_filtered dictionary
    region_filtered = filter_large_regions(json_dict, filter_size)
    # print(region_filtered.keys())

    # Create a checklist of patches
    visited = [[False] * len(img_patches[0]) for _ in range(len(img_patches))]

    # Create a reconstruction guide from dict patches
    img_recon = [[['background', -1]] * len(img_patches[0]) for _ in range(len(img_patches))]

    # Add ref patches to the instance dict and img_recon, check visited list
    for row in range(len(img_patches)):
        for col in range(len(img_patches[row])):
            patch_properties = img_patches[row][col]
            top, left = patch_properties[0], patch_properties[1]

            img_patch = img[top:top+patch_size, left:left+patch_size]
            seg_patch = mask[top:top+patch_size, left:left+patch_size]

            most_frequent_value = count_most_frequent_pixel(seg_patch, threshold)
            for region in region_filtered.keys():
                if most_frequent_value == region_filtered[region]['value'] and visited[row][col] is False:
                    region_filtered[region]['patches'].append(img_patch)
                    img_recon[row][col] = [region, len(region_filtered[region]['patches']) - 1]
                    visited[row][col] = True

    # Add background and non-ref patches, call 'background' together
    region_filtered['background'] = []

    # Add the non-ref patches to background and img_recon
    for row in range(len(img_patches)):
        for col in range(len(img_patches[row])):
            if visited[row][col] is False:
                patch_properties = img_patches[row][col]
                top, left = patch_properties[0], patch_properties[1]
                img_patch = img[top:top+patch_size, left:left+patch_size]

                region_filtered['background'].append(img_patch)
                img_recon[row][col] = ['background', len(region_filtered['background']) - 1]
                visited[row][col] = True
    
    return region_filtered, img_recon


""" NOTE: parameters to choose:
        - division - divide_into_patches: affect the total number of patches and patch size
        - threshold - check_segmentation_region: affect the number of reference patches of each instance
        - filter_size - filter_large_regions: affect the number of instance regions for reference
"""

img = cv2.imread("sample/0039.png")
mask = cv2.imread('sample/mask.png', cv2.IMREAD_GRAYSCALE)

img_patches, patch_size, pad = divide_into_patches(img, division=6)

# Padding the image to fit the patch division
mask_pad = (pad[0], pad[1])
img = np.pad(img, pad, mode='constant')
mask = np.pad(mask, mask_pad, mode='constant')

cv2.imwrite('mask_pad.png', mask)

f = open('sample/label.json',)
json_dict = json.load(f)
f.close()

region_filtered, img_recon =  patch_partition(img_patches, json_dict, img, mask, patch_size, threshold=(patch_size**2 * 0.2))

recon_lst = []
for j in range(len(img_recon)):
    list_patches_row = []
    for i in img_recon[j]:
        if i[0] == 'background':
            list_patches_row.append(region_filtered['background'][i[1]])
        else:
            s = region_filtered[i[0]]["patches"][i[1]]
            list_patches_row.append(s)
    recon = np.concatenate(list_patches_row, axis=1)
    recon_lst.append(recon)
recon = np.concatenate(recon_lst, axis=0)
cv2.imshow("a", recon)
cv2.waitKey(0)
cv2.destroyAllWindows()

for i in img_recon:
    print(i)

# print(patch_size)
# print()
# cv2.imshow("a", region_filtered['background'][42])
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# TODO: write code to get the reference image generated.
# TODO: implement reference selection for patches (Cosine similarity)
