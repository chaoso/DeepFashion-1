import numpy as np
def allAttributes(img_ID, Data_G2, Data_matlab, skip):
    # ----- Get all attributes from image ID -----
    # Input:    Image ID
    #
    #           G2 dataset: Assumed to have been loaded
    #           with the "h5py" library first
    #
    #           Data_matlab: Complete "language_original.mat" file.
    #           Assumed to have been loaded with with scipy.io's loadmat.
    #
    # Output:   Return a vector with all attributes (len=50)

    # Get image from ID
    segment_IMG = Data_G2['b_'][img_ID][0].T
    img_mean = Data_G2['ih_mean']
    color_IMG = Data_G2['ih'][img_ID] + img_mean

    # ---------- Matlab attributes ----------
    def MatlabAttributes(imgID, Data_matlab):
        gender = int(Data_matlab.get('gender_')[imgID])
        #color = int(mat.get('color_')[imgID])
        return gender#, color

    def textVect(imgID, Data_matlab):
        textIdx = Data_matlab.get('codeJ')[imgID][0]
        wordVect = np.zeros(1000) # Pre-difined length
        for idx in textIdx:
            wordVect[idx] += 1
        return wordVect

    # ---------- Create own attributes ----------
    # Height
    def findHeight(segment_image):
        a = np.argmax(segment_image, 1)
        a = np.trim_zeros(a)
        return len(a)

    # Width
    def findWidth(segment_image):
        a = np.argmax(segment_image, 0)
        a = np.trim_zeros(a)
        return len(a)

    # Width to height ratio
    def ratio(segment_image):
        try:
            value = findWidth(segment_image)/findHeight(segment_image)
        except:
            value = 0.461515512910306 # mean ratio. Because some missing values
        return value

    # Long hair
    def hasLongHair(segment_image):
        # Hair segmentation map
        hair = np.where(segment_image != 1, 0, segment_image)

        # Torso segmentation map
        torso = np.where(segment_image != 3, 0, segment_image)

        # Lowest pixel of hair
        hair_low = np.argmax(hair, 1)
        hair_low = np.argwhere(hair_low != 0)
        if hair_low.size == 0:
            return 0
        hair_low = hair_low[-1]

        # Highest pixel of torso
        torso_high = np.argmax(torso, 1)
        torso_high = np.argwhere(torso_high != 0)

        if torso_high.size == 0:
            return 0

        torso_high = torso_high[0]

        # Has long hair
        if hair_low < torso_high:
            hasLongHair = 0
        else:
            hasLongHair = 1
        return hasLongHair

    def RGB_Y_mean(segment_IMG, color_IMG):
        # Find all pixels with skin colors
        face = np.where((segment_IMG != 2), 0, segment_IMG)
        leg = np.where((segment_IMG != 5), 0, segment_IMG)
        arms = np.where((segment_IMG != 6), 0, segment_IMG)
        allSkin = face + leg + arms

        # make segmentation binary
        allSkin = np.where((allSkin != 0), 1, allSkin)

        # Include only pixels with skin
        img_R = color_IMG[0]
        img_R = np.where((allSkin != 1), 0, img_R)

        img_G = color_IMG[1]
        img_G = np.where((allSkin != 1), 0, img_G)

        img_B = color_IMG[2]
        img_B = np.where((allSkin != 1), 0, img_B)

        img_Y = (img_R + img_G + img_B)/3

        # Choose median (True) or mean values (False)
        medianColor = True
        
        # Function used to calculate the median value
        def median(img):
            img_median = img[img != 0]
            img_median.sort()
            return np.median(img_median)
        
        # Mean colors
        if medianColor == False:
            R_mean = img_R.mean()
            G_mean = img_G.mean()
            B_mean = img_B.mean()
            Y_mean = img_Y.mean()
            return R_mean, G_mean, B_mean, Y_mean
        
        # Median colors
        elif medianColor == True:
            R_median = median(img_R)
            G_median = median(img_G)
            B_median = median(img_B)
            Y_median = median(img_Y)
            return R_median, G_median, B_median, Y_median

    # Get color attributes
    R_mean, G_mean, B_mean, Y_mean = RGB_Y_mean(segment_IMG, color_IMG)

    if skip == True:
        humanAttr = np.array([int(img_ID), 0, 0, 0, 0, 0, 0, 0, 0, MatlabAttributes(img_ID, Data_matlab)])
    else:
        humanAttr = np.array([int(img_ID), findHeight(segment_IMG), findWidth(segment_IMG), 
                              ratio(segment_IMG), hasLongHair(segment_IMG), R_mean, G_mean, B_mean, Y_mean, MatlabAttributes(img_ID, Data_matlab)])
    
    #wordAttr = textVect(img_ID, Data_matlab)

    #allAttr = list(humanAttr) + list(wordAttr)

    return list(humanAttr)#allAttr[:50] # Works as dummy for now