import pandas as pd
import mysql.connector
from dateutil.relativedelta import relativedelta
from datetime import datetime
from decimal import Decimal
from catboost import CatBoostRegressor
from sklearn import ensemble
from sklearn.tree import DecisionTreeClassifier
import ConfigParser
import subprocess
import string

pd.options.display.max_columns = None

Config = ConfigParser.ConfigParser()
Config.read("properties.ini")

cmd='date +"%Y-%m"'
prediction_month = string.strip(subprocess.check_output(cmd, shell=True))

conn = mysql.connector.connect(host= Config.get('DEFAULT', 'host'),
                port=Config.get('DEFAULT', 'port'), user=Config.get('DEFAULT', 'user'), 
                passwd=Config.get('DEFAULT', 'passwd'), 
                db=Config.get('DEFAULT', 'db'))

query_string = "SELECT distinct cr.user_id as user_id, cr.id as id, cr.count_months as count_months, cr.amount as amount, cr.month_payment as month_payment,inv.amount as payed_this_month, cps.left_amount as to_pay, cps.amount as planned, cr.created_at as created_at, cps.date as date FROM credit cr join credit_payment_schedule cps on cr.id = cps.credit_id left join (select credit_id, sum(amount)as amount from invoice where created_at > '{}' AND created_at < '{}' group by credit_id) inv on cr.id=inv.credit_id WHERE (cr.created_at > '2019-12-01' AND cr.created_at < '{}') and (cps.date > '{}' AND cps.date < '{}' ) and cr.status in (2) and (cr.user_id > 19) and cr.user_id not in (23,24,26,28,33)"
insertion_string = "INSERT INTO credit_prediction(month, planned, left_to_pay, prediction) VALUES('{}',{},{},{})"

column_names = ['user_id', 'id', 'count_months', 'amount', 'month_payment',
       'payed_this_month', 'to_pay', 'planned', 'created_at', 'date']
train = pd.DataFrame(columns=column_names)


date_to = (datetime.strptime(prediction_month, "%Y-%m").date()).strftime("%Y-%m-%d")

while (date_to>'2020-01-01'):
    date_from = (datetime.strptime(date_to, "%Y-%m-%d").date() + relativedelta(months=-1)).strftime("%Y-%m-%d")
    train_query = pd.read_sql_query(query_string.format(date_from, date_to, date_from, date_from, date_to), conn)
    train = train.append(train_query)
    date_to = (datetime.strptime(date_to, "%Y-%m-%d").date() + relativedelta(months=-1)).strftime("%Y-%m-%d")
train = train.fillna(0)

test_from = (datetime.strptime(prediction_month, "%Y-%m").date()).strftime("%Y-%m-%d")
test_to = (datetime.strptime(test_from, "%Y-%m-%d").date() + relativedelta(months=+1)).strftime("%Y-%m-%d")
test = pd.read_sql_query(query_string.format(test_from, test_to, test_from, test_from, test_to), conn)
test = test.fillna(0)

#predictions
features = ['count_months', 'amount', 'month_payment']

X_train = train[features]
X_val = test[features]
y_train = train.payed_this_month.astype('int64').copy()
y_val = test.payed_this_month.astype('int64').copy()

model=CatBoostRegressor(iterations=200, depth=5, learning_rate=0.1, loss_function='RMSE')
model.fit(X_train, y_train)
preds = model.predict(X_val)
pred_catboost = sum(preds)

clf = ensemble.RandomForestClassifier(n_estimators=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)
pred_randforest = sum(y_pred)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
proba = model.predict(X_val)
pred_destree = sum(proba)

#writing to db

Planned = sum(test.planned)
Payed = sum(test.payed_this_month)
left = sum((test.to_pay).astype(float))
prediction = (pred_catboost+pred_destree+pred_randforest)/3

query = insertion_string.format(datetime.strptime(prediction_month, "%Y-%m").date(), Decimal(Planned), Decimal(left), Decimal(prediction))
cursor = conn.cursor()
cursor.execute(query)
conn.commit()
conn.close()