import pandas as pd
import matplotlib.pyplot as plt


class TitanicAnalytics(object):
    """
    Analysis of the survival of Titanic passengers
    train.csv - the data set that will be used to build the model (training sample)
    Explanations on the fields of the data set:
        PassengerId - passenger ID
        Survival - the field in which the person is saved (1) or not (0)
        Pclass - contains socio-economic status:
         1. High
         2. Medium
         3. Low
        Name - passenger's name
        Sex - passenger sex
        Age - age
        SibSp - contains information on the number of relatives of the 2nd order (husband, wife, brothers, sisters)
        Parch - contains information on the number of relatives board of the first order (mother, father, children)
        Ticket - ticket number
        Fare - ticket price
        Cabin - cabin
        Embarked - landing port
         • C - Cherbourg
         • Q - Queenstown
         • S - Southampton
    """

    def __init__(self, file_path='./data/train.csv', prepare=False):
        self._data = pd.read_csv(file_path, sep=',')
        if prepare:
            self.prepare_data()

    def calculate_by_classes(self):
        """
        The number of survivors and drowned in the context of classes
        """
        _res = self._data.pivot_table('PassengerId', 'Pclass', 'Survived', 'count')
        return _res.plot(kind='bar', stacked=True)

    def calculate_by_relations(self):
        """
        The influence of the number of relatives on the fact of salvation
        """
        fig, axes = plt.subplots(ncols=2)
        self._data.pivot_table('PassengerId', ['SibSp'], 'Survived', 'count').plot(ax=axes[0],
                                                                                   title='SibSp')
        self._data.pivot_table('PassengerId', ['Parch'], 'Survived', 'count').plot(ax=axes[1],
                                                                                   title='Parch')

    @property
    def cabin_count(self):
        return self._data.PassengerId[self._data.Cabin.notnull()].count()

    @property
    def age_count(self):
        return self._data.PassengerId[self._data.Age.notnull()].count()

    def prepare_data(self):
        self.fill_embarked_data()
        self.fill_age_data()
        self.drop_unnecessary_columns()

    def fill_age_data(self):
        self._data.Age = self._data.Age.median()

    def fill_embarked_data(self):
        max_pass_embarked = self._data.groupby('Embarked').count()['PassengerId']
        self._data.Embarked[self._data.Embarked.isnull()] = max_pass_embarked.idxmax()

    def drop_unnecessary_columns(self):
        self._data = self._data.drop(['Name', 'Ticket', 'Cabin'], axis=1)

    @staticmethod
    def show():
        plt.show()

    def solve_0_variant(self):
        """Estimate male passengers from the first class"""
        return self._data[(self._data.Sex == 'male') & (self._data.Pclass == 1)]

    def solve_1_variant(self):
        """Estimate the number of children from the second class"""
        return self._data[(self._data.Age <= 18) & (self._data.Pclass == 2)].PassengerId.count()

    def solve_2_variant(self):
        """Estimate the number of single passengers (without relatives)"""
        return self._data[(self._data.SibSp == 0) & (self._data.Parch == 0)].PassengerId.count()

    def solve_3_variant(self):
        """Estimate the number of passengers who have boarded the Queenstown port with expensive tickets"""
        return self._data[(self._data.Embarked == 'S') & (self._data.Pclass == 1)].PassengerId.count()

    def solve_4_variant(self):
        """Estimate the average age of female passengers"""
        return self._data[(self._data.Sex == 'female')].Age.mean()

    def solve_5_variant(self):
        """Estimate the number of single elderly people"""
        return self._data[(self._data.SibSp == 0) &
                          (self._data.Parch == 0) &
                          (self._data.Age >= 60)].PassengerId.count()

    def solve_6_variant(self):
        """Show statistics on children"""
        return self._data[self._data.Age < 18]

    def solve_7_variant(self):
        """Estimate the average ticket price for each port"""
        return self._data.groupby(['Embarked']).Fare.mean()

    def solve_8_variant(self):
        """Estimate the average ticket price for each social class"""
        return self._data.groupby(['Pclass']).Fare.mean()


if __name__ == "__main__":
    analytics = TitanicAnalytics()
    for variant in range(9):
        print(f"{'*' * 16}_Variant_№_{variant + 1}_{'*' * 16}")
        print(getattr(analytics, f"solve_{variant}_variant")())
