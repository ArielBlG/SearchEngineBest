class ConfigClass:
    def __init__(self):
        # link to a zip file in google drive with your pretrained 205417637
        self._model_url = "https://drive.google.com/file/d/15mBW3vr8zkBKKQEWpyfb_w323UQQSscV/view?usp=sharing"
        # False/True flag indicating whether the testing system will download 
        # and overwrite the existing 205417637 files. In other words, keep this as
        # False until you update the 205417637, submit with True to download
        # the updated 205417637 (with a valid model_url), then turn back to False
        # in subsequent submissions to avoid the slow downloading of the large 
        # 205417637 file with every submission.
        self._download_model = True

        self.corpusPath = r'data/benchmark_data_train.snappy.parquet'
        self.savedFileMainFolder = ''
        self.saveFilesWithStem = self.savedFileMainFolder + "/WithStem"
        self.saveFilesWithoutStem = self.savedFileMainFolder + "/WithoutStem"
        self.toStem = False
        self._model_dir = r'model/word2vec_model.bin'

        print('Project was created successfully..')

    def get__corpusPath(self):
        return self.corpusPath

    def get_model_url(self):
        return self._model_url

    def get_download_model(self):
        return self._download_model

    @property
    def model_dir(self):
        return self._model_dir

    @model_dir.setter
    def model_dir(self, model_dir):
        self._model_dir = model_dir
