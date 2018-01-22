import numpy as np
from numpy.matlib import repmat
from numpy.linalg import eig
from PIL import Image
import matplotlib.pyplot as plt
import os

class face_recognize():
    def __init__(self, Face_data_dir = None):
        if Face_data_dir is None:
            print("Face data stored in current dir ")
            self.data_dir = os.getcwd()+"/FACESdata"
        else:
            print("Face data stored in dir {}".format(Face_data_dir))
            self.data_dir = Face_data_dir + "/FACESdata"
        self.ori_shape  = np.array(Image.open(self.data_dir+"/s1/1.pgm").convert('L')).shape

    # convert image to vector
    def img2vec(self, img_path):
        img = Image.open(img_path).convert('L')
        # reshape img matrix into a column
        return np.array(img).reshape(self.ori_shape[0]*self.ori_shape[1], 1)

    def read_data(self, plot_mean_face = None):
        # read all face data
        print("Assuming there are 40 faces and 10 pic for each face. Need to change reading config is not satisfied")
        data = [self.img2vec(img_path = (self.data_dir+"/s" + str(i) + "/" + str(j) + ".pgm"))
                for i in range(1, 41) for j in range(1, 11)]
        # save data into face matrix
        self.face_mat = np.concatenate(data, axis=1)
        # get mean face matrix
        self.mean_face_mat = np.mean(self.face_mat, axis=1).reshape(self.ori_shape[0] * self.ori_shape[1], 1)
        # plot the average face
        # default setting: plot and save the mean face
        plot_mean_face = 1 if plot_mean_face is None else 0
        if plot_mean_face == 1:
            mean_face_img = Image.fromarray(np.uint8(self.mean_face_mat.reshape(self.ori_shape)), 'L')
            mean_face_img.show()
            mean_face_img.save("MeanFace.jpg")

        # Normalize face matrix
        self.norm_mat = self.face_mat - repmat(self.mean_face_mat, 1, self.face_mat.shape[1])

        print("Norm Matrix:")
        print(self.norm_mat)

    # need save top k eig vector into egi face
    def egi_face(self, k):
        # cov_mat = np.matmul(self.norm_mat.transpose(), self.norm_mat)
        eig_vals, eig_vects = eig(np.matmul(self.norm_mat.transpose(), self.norm_mat))
        # sort based on eig value from max to min
        sorted_idx = np.argsort(eig_vals)[::-1]
        eig_fac = eig_vects[sorted_idx[:k]]
        self.eigenface_matrix = np.matmul(self.norm_mat,
                                          np.concatenate([vect.reshape(-1, 1) for vect in eig_fac], axis=1))

        print("EIGENFACES:")
        print(self.eigenface_matrix)
        print("Eigenface matrix shape = .{}".format(self.eigenface_matrix.shape))
        # get weight of data projection on Eigenface
        self.weight_mat_database = self.proj_wt(self.norm_mat)


    # plot top k Eigenface
    def plot_top_k_egi_face(self, k):
        print("plotting top {} Eigenface".format(k))
        for i in range(k):
            self.norm_vect2face_plt(self.eigenface_matrix[:, i], "Eigenface_"+str(i+1))

    # function to plot face based on norm vector input
    def norm_vect2face_plt(self, vec, fig_name):
        fig = plt.figure(fig_name)
        plt.imshow(np.add(vec.reshape(self.ori_shape), self.mean_face_mat.reshape(self.ori_shape)),
                   cmap=plt.get_cmap('Greys'))
        plt.savefig(fig_name+".jpg")
        plt.show()
        


    # function to get weight of projection
    """
    projection of vector u onto vector v = (dot(u,v)/2norm(v))v
    the weight is dot(u,v)/2norm(v)
    """
    def proj_wt(self, input_mat):
        dot_u_v = np.dot(input_mat.transpose(), self.eigenface_matrix)
        v_2norm = sum(self.eigenface_matrix**2)
        return np.divide(dot_u_v, v_2norm)

    # find the face in FACESdata based on its index in the matrix
    # this function is still under construction
    def idx2face(self, idx):
        s_num = idx//10 + 1
        pic_num = (idx+1)%10
        if pic_num == 0:
            pic_num = 10
        face_img_path = self.data_dir + "/s"+str(s_num)+"/"+str(pic_num)+".pgm"
        print("The face image path = {}".format(face_img_path))
        print("Show the face")
        img_predict = Image.open(face_img_path)
        img_predict.show()
        img_predict.save("PREDICTED_Image.jpg")
        return face_img_path

    def face_predict(self, test_vec):
        # normalize test vec
        vec = test_vec.reshape(self.ori_shape[0]*self.ori_shape[1], 1) - self.mean_face_mat
        # get weight for test vector
        test_wt = self.proj_wt(vec)
        # compute Euclidean dis among database
        Euclidean_distance_sq = sum((self.weight_mat_database-test_wt).transpose()**2)
        # output the one with smallest Euclidean dis
        idx = np.argmin(Euclidean_distance_sq)
        print(idx)
        return self.idx2face(idx)


if __name__ == "__main__":
    fr = face_recognize()
    fr.read_data()
    k = input("Number of eigenface k = ")
    fr.egi_face(int(k))
    fr.plot_top_k_egi_face(5)
    # test
    img_path = 'TEST_Image.pgm'
    Image.open(img_path).save("TEST_Image.jpg")
    test_vec = fr.img2vec(img_path)
    fr.face_predict(test_vec)


