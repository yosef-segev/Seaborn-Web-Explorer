import seaborn as sns

class DataService:
    def __init__(self, dataset_name="titanic"):
        # טעינת הנתונים פעם אחת ושמירתם בשדה (field) של המחלקה
        self.df = sns.load_dataset(dataset_name)
        self.dataset_name = dataset_name

    def get_df(self):
        # פונקציה שחשופת את ה-DataFrame החוצה
        return self.df