from anchor import anchor_base
import numpy as np
import sklearn
import image_utils
import csgm
import torch
import dcgan
import svgd
import matplotlib.pyplot as plt
class AnchorImageMNIST(object):
    """bla"""
    def __init__(self, distribution_path=None,
                 transform_img_fn=None, n=1000, dummys=None, white=None,
                 segmentation_fn=None, G=None, dataset = None, batch_norm=False,
                 encode = False, encoder=None, threshold=0.05, true_sampling=False):
        """"""
        self.hide = True
        self.white = white
        self.batch_norm = batch_norm
        self.threshold = threshold
        self.true_sampling = true_sampling
        self.encode = encode
        # generator
        if G is not None:
            self.G = G(0).cuda()
            if batch_norm:
                self.G.train()
            else:
                self.G.eval()
        if encode:
            # assumes encoder is a function that loads the encoder
            self.encoder = encoder('cpu')
            self.encode = encode
        if self.true_sampling:
            # to compute "truer density sampling"
            def sample_p(num_samples):
                x = np.random.normal(loc=self.threshold, scale=self.threshold/6, size=(num_samples))
                x = x[x < self.threshold * 2]
                x = x[x > 0]
                x[x > self.threshold] = self.threshold * 2 - x[x > self.threshold]
                return x
            self.sample_p = sample_p
        if segmentation_fn is None:
            segmentation_fn = lambda x: image_utils.create_segments(x) 
        self.segmentation = segmentation_fn
        if dummys is not None:
            self.hide = False
            self.dummys = dummys
        elif distribution_path:
            self.hide = False
            import os
            import skimage

            if not transform_img_fn:
                def transform_img(path):
                    img = skimage.io.imread(path)
                    short_egde = min(img.shape[:2])
                    yy = int((img.shape[0] - short_egde) / 2)
                    xx = int((img.shape[1] - short_egde) / 2)
                    crop_img = img[yy: yy + short_egde, xx: xx + short_egde]
                    return skimage.transform.resize(crop_img, (224, 224))

                def transform_imgs(paths):
                    out = []
                    for i, path in enumerate(paths):
                        if i % 100 == 0:
                            print(i)
                        out.append(transform_img(path))
                    return out
                transform_img_fn = transform_imgs
            all_files = os.listdir(distribution_path)
            all_files = np.random.choice(
                all_files, size=min(n, len(all_files)), replace=False)
            paths = [os.path.join(distribution_path, f) for f in all_files]
            self.dummys = transform_img_fn(paths)
        elif dataset is not None:
            self.mnist = dataset
            # assumes original dataset is from [0,1]
            self.get_target = lambda x, mask : (x - 0.5) * 2 * mask
            self.present = None
    def get_sample_fn(self, image, classifier_fn, lime=False):
        import copy
        # segments = slic(image, n_segments=100, compactness=20)
        segments = self.segmentation(image)
        fudged_image = image.copy()
        #for x in np.unique(segments):
        #    fudged_image[segments == x] = (np.mean(image[segments == x][:, 0]),
        #                                   np.mean(image[segments == x][:, 1]),
        #                                   np.mean(image[segments == x][:, 2]))
        #if self.white is not None:
        #    fudged_image[:] = self.white
        features = list(np.unique(segments))
        n_features = len(features)

        true_label = np.argmax(classifier_fn(np.expand_dims(image, 0))[0])
        print ('True pred', true_label)

        def lime_sample_fn(num_samples, batch_size=50):
            # data = np.random.randint(0, 2, num_samples * n_features).reshape(
            #     (num_samples, n_features))
            data = np.zeros((num_samples, n_features))
            labels = []
            imgs = []
            sizes = np.random.randint(0, n_features, num_samples)
            all_features = range(n_features)
            # for row in data:
            for i, size in enumerate(sizes):
                row = np.ones(n_features)
                chosen = np.random.choice(all_features, size)
                # print chosen, size,
                row[chosen] = 0
                data[i] = row
                # print row
                temp = copy.deepcopy(image)
                zeros = np.where(row == 0)[0]
                mask = np.zeros(segments.shape).astype(bool)
                for z in zeros:
                    mask[segments == z] = True
                temp[mask] = fudged_image[mask]
                imgs.append(temp)
                if len(imgs) == batch_size:
                    preds = classifier_fn(np.array(imgs))
                    labels.extend(preds)
                    imgs = []
            if len(imgs) > 0:
                preds = classifier_fn(np.array(imgs))
                labels.extend(preds)
            # return imgs, np.array(labels)
            return data, np.array(labels)

        if lime:
            return segments, lime_sample_fn



        def sample_fn_dummy(present, num_samples, compute_labels=True):
            if not compute_labels:
                data = np.random.randint(
                    0, 2, num_samples * n_features).reshape(
                        (num_samples, n_features))
                data[:, present] = 1
                return [], data, []
            data = np.zeros((num_samples, n_features))
            # data = np.random.randint(0, 2, num_samples * n_features).reshape(
            #     (num_samples, n_features))
            if len(present) < 5:
                data = np.random.choice(
                    [0, 1], num_samples * n_features, p=[.8, .2]).reshape(
                        (num_samples, n_features))
            data[:, present] = 1
            chosen = np.random.choice(range(len(self.dummys)), data.shape[0],
                                      replace=True)
            labels = []
            imgs = []
            for d, r in zip(data, chosen):
                temp = copy.deepcopy(image)
                zeros = np.where(d == 0)[0]
                mask = np.zeros(segments.shape).astype(bool)
                for z in zeros:
                    mask[segments == z] = True
                if self.white:
                    temp[mask] = 1
                else:
                    temp[mask] = self.dummys[r][0][mask]
                imgs.append(temp)
                # pred = np.argmax(classifier_fn(temp.to_nn())[0])
                # print self.class_names[pred]
                # labels.append(int(pred == true_label))
            # import time
            # a = time.time()
            imgs = np.array(imgs)
            preds = classifier_fn(imgs)
            # print (time.time() - a) / preds.shape[0]
            imgs = []
            preds_max = np.argmax(preds, axis=1)
            labels = (preds_max == true_label).astype(int)
            raw_data = np.hstack((data, chosen.reshape(-1, 1)))
            return raw_data, data, np.array(labels)

        # testing with kde method
        def sample_fn_stein(present, num_samples, compute_labels=True):
            '''
            present = which segments to choose from...
            '''
            data = np.zeros((num_samples,n_features))
            data[:, present] = 1
            print(num_samples)
            # now generate some images
            _, mask = image_utils.create_mask(None, segments, {'feature': present})
            target = self.get_target(image, mask)
            # first time we called it
            if self.stein is None:
                # setup for stein's
                G = dcgan.ProbGenerator(self.G, mask, target)
                K = svgd.RBF()
                X = torch.randn(num_samples,100).cuda()
                adam = torch.optim.Adam([X], lr=1e-1)
                self.stein = svgd.SVGD(G,K,adam)
                # train...
                print("Starting Stein Training")
                X = self.stein.train(X, num_iter = 1000)
                print("Trained!")
                 
            BS = 64
            collected = 0
            labels = np.zeros((num_samples)).astype(int)
            raw_data = data
            print("Predicting...")
            while collected < num_samples:
                X = self.stein.sample(BS)
                if X.shape[0] == 0:
                #    print("Bad samples")
                    continue
                #print("KDE found", X.shape[0], "good samples.") 
                backgrounds = X.view(-1,28,28).data.cpu().numpy() * (1-target)
                end = min(backgrounds.shape[0], num_samples - collected)
                backgrounds = backgrounds[:end]
                current_batch = backgrounds + target 
                current_preds = classifier_fn(current_batch)
                current_preds_max = np.argmax(current_preds, axis=1)
                current_labels = (current_preds_max == true_label).astype(int)
                labels[collected:collected + end] = current_labels
                collected += end
            print("Collected", num_samples,"!")
            return raw_data, data, labels

        # new and IMPROVED with csgm + KDE
        def sample_fn_csgm_kde(present, num_samples, compute_labels=True):
            '''
            present = which segments to choose from...
            '''
            data = np.zeros((num_samples,n_features))
            data[:, present] = 1
            print(num_samples)
            # now generate some images
            _, mask = image_utils.create_mask(None, segments, {'feature': present})
            target = self.get_target(image, mask)
            if self.present != present:
                self.sampler = csgm.CSGM(target,mask,self.G,num_samples,bandwidth=0.5)
            self.present = present
            BS = 64
            #raw_data = np.zeros((num_samples,28,28))
            raw_data = data
            _,_,backgrounds = self.sampler.sample(num_samples)
            current_batch = backgrounds + target
            current_preds = classifier_fn(current_batch)
            current_preds_max = np.argmax(current_preds, axis=1)
            labels = (current_preds_max == true_label).astype(int)
            return raw_data, data, labels

        def sample_fn_csgm_encoder(present, num_samples, compute_labels=True):
            '''
            present = which segments to choose from...
            '''
            if not compute_labels:
                data = np.random.randint(
                    0, 2, num_samples * n_features).reshape(
                        (num_samples, n_features))
                data[:, present] = 1
                return [], data, []

            data = np.zeros((num_samples,n_features))
            data[:, present] = 1
            # now generate some images 
            _, mask = image_utils.create_mask(None, segments, {'feature': present})
            target = self.get_target(image, mask)
            
            input_X = (mask*image).reshape(-1,1,28,28)
            Zs = self.encoder(torch.from_numpy(input_X).type(torch.FloatTensor))
            
            #plt.imshow(target)
            BS = 64
            #raw_data = np.zeros((num_samples,28,28))
            raw_data = data
            labels = np.zeros((num_samples)).astype(int)
            for j in range(0,num_samples,BS):
                n_s = min(num_samples,j+BS) - j
                if self.true_sampling:
                    _,_,backgrounds = csgm.reconstruct_batch_threshold(target, mask, np.sum(mask), self.G, n_s,
                                                             self.sample_p(n_s), 
                                                             lr= 1e-1 if self.batch_norm else 1e-2, 
                                                             init_mu = Zs)
                else:
                    _,_, backgrounds = csgm.reconstruct_batch(target, mask, np.sum(mask), self.G, n_s, 
                                                             lr= 1e-1 if self.batch_norm else 1e-2, 
                                                             init_mu = Zs,
                                                             threshold = self.threshold)
#                raw_data[j:j+n_s] = raw_data_.squeeze()
                current_batch = np.zeros((n_s, 28,28))
                for i in range(len(backgrounds)):
                    temp = copy.deepcopy(target)
                    curr = backgrounds[i] ## should be (28,28)
                    temp += curr
                    current_batch[i] = np.expand_dims(temp,0)
                current_preds = classifier_fn(current_batch)
                current_preds_max = np.argmax(current_preds, axis=1)
                current_labels = (current_preds_max == true_label).astype(int)
                labels[j:j+n_s] = current_labels 
            return raw_data, data, labels             
    
        # new and IMPROVED with csgm
        def sample_fn_csgm(present, num_samples, compute_labels=True):
            '''
            present = which segments to choose from...
            '''
            if not compute_labels:
                data = np.random.randint(
                    0, 2, num_samples * n_features).reshape(
                        (num_samples, n_features))
                data[:, present] = 1
                return [], data, []

            data = np.zeros((num_samples,n_features))
            data[:, present] = 1
            # now generate some images 
            _, mask = image_utils.create_mask(None, segments, {'feature': present})
            target = self.get_target(image, mask)
            #plt.imshow(target)
            BS = 64
            #raw_data = np.zeros((num_samples,28,28))
            raw_data = data
            labels = np.zeros((num_samples)).astype(int)
            for j in range(0,num_samples,BS):
                n_s = min(num_samples,j+BS) - j
                if self.true_sampling:
                    _,_,backgrounds = csgm.reconstruct_batch_threshold(target, mask, np.sum(mask), self.G, n_s, self.sample_p(n_s), 
                                                             lr= 1e-1 if self.batch_norm else 1e-2) 
                else:
                    _,_, backgrounds = csgm.reconstruct_batch(target, mask, np.sum(mask), self.G, n_s, 
                                                             lr= 1e-1 if self.batch_norm else 1e-2, 
                                                             threshold = self.threshold)
                current_batch = np.zeros((n_s, 28,28))
                for i in range(len(backgrounds)):
                    temp = copy.deepcopy(target)
                    curr = backgrounds[i] ## should be (28,28)
                    temp += curr
                    current_batch[i] = np.expand_dims(temp,0)
                current_preds = classifier_fn(current_batch)
                current_preds_max = np.argmax(current_preds, axis=1)
                current_labels = (current_preds_max == true_label).astype(int)
                labels[j:j+n_s] = current_labels 
            return raw_data, data, labels             
 
        def sample_fn(present, num_samples, compute_labels=True):
            # TODO: I'm sampling in this different way because the way we were
            # sampling confounds size of the document with feature presence
            # (larger documents are more likely to have features present)
            data = np.random.randint(0, 2, num_samples * n_features).reshape(
                (num_samples, n_features))
            data[:, present] = 1
            if not compute_labels:
                return [], data, []
            imgs = []
            for row in data:
                temp = copy.deepcopy(image)
                zeros = np.where(row == 0)[0]
                mask = np.zeros(segments.shape).astype(bool)
                for z in zeros:
                    mask[segments == z] = True
                temp[mask] = fudged_image[mask]
                imgs.append(temp)
            preds = classifier_fn(np.array(imgs))
            preds_max = np.argmax(preds, axis=1)
            labels = (preds_max == true_label).astype(int)
            # raw_data = imgs
            raw_data = data
            return raw_data, data, labels
        self.stein = None
#        sample = sample_fn_stein
        sample = sample_fn_csgm_encoder if self.encode else ( sample_fn_csgm if self.hide else sample_fn_dummy )
        return segments, sample

    def explain_instance(self, image, classifier_fn, threshold=0.95,
                          delta=0.1, tau=0.15, batch_size=100,
                           **kwargs):
        # classifier_fn is a predict_proba
        segments, sample = self.get_sample_fn(image, classifier_fn) 
        best_coverage, exp = anchor_base.AnchorBaseBeam.anchor_beam(
            sample, delta=delta, epsilon=tau, batch_size=batch_size,
            desired_confidence=threshold, coverage_samples=100,max_anchor_size=3,**kwargs)
        return best_coverage, segments, self.get_exp_from_hoeffding(image, exp)

    def get_exp_from_hoeffding(self, image, hoeffding_exp):
        """
        bla
        """
        ret = []

        features = hoeffding_exp['feature']
        means = hoeffding_exp['mean']
        if 'negatives' not in hoeffding_exp:
            negatives_ = [np.array([]) for x in features]
        else:
            negatives_ = hoeffding_exp['negatives']
        for f, mean, negatives in zip(features, means, negatives_):
            train_support = 0
            name = ''
            if negatives.shape[0] > 0:
                unique_negatives = np.vstack({
                    tuple(row) for row in negatives})
                distances = sklearn.metrics.pairwise_distances(
                    np.ones((1, negatives.shape[1])),
                    unique_negatives)
                negative_arrays = (unique_negatives
                                   [np.argsort(distances)[0][:4]])
                negatives = []
                for n in negative_arrays:
                    negatives.append(n)
            else:
                negatives = []
            ret.append((f, name, mean, negatives, train_support))
        return ret
