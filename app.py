import flask
import pickle
import pandas as pd
import sklearn


lm_model = 'lm.pkl'
lm=pickle.load(open(lm_model, 'rb'))




rf_model = 'rf.pkl'
rf=pickle.load(open(rf_model, 'rb'))



app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def main():
    input_variables = pd.DataFrame(columns=['player','inning','against','home','toss_winner'])
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':
        
        index=0
        team=flask.request.form['team1']
        against = flask.request.form['team2']
        home=1
        selection = flask.request.form['inning']
        toss=flask.request.form['toss']
        if toss=='team1':
            toss_winner=1
            if selection=='batting':
                inning=1
            else :
                inning=2
        else :
            toss_winner=0
            if selection=='batting':
                inning=2
            else :
                inning=1
        
        player = flask.request.form['player1']
    
        input_variables.loc[index,'player']=player
        input_variables.loc[index,'inning']=inning
        input_variables.loc[index,'against']=against
        input_variables.loc[index,'home']=home
        input_variables.loc[index,'toss_winner']=toss_winner
        index=index+1
    
        player = flask.request.form['player2']
    
        input_variables.loc[index,'player']=player
        input_variables.loc[index,'inning']=inning
        input_variables.loc[index,'against']=against
        input_variables.loc[index,'home']=home
        input_variables.loc[index,'toss_winner']=toss_winner
        index=index+1
        player = flask.request.form['player3']
    
        input_variables.loc[index,'player']=player
        input_variables.loc[index,'inning']=inning
        input_variables.loc[index,'against']=against
        input_variables.loc[index,'home']=home
        input_variables.loc[index,'toss_winner']=toss_winner
        index=index+1
        player = flask.request.form['player4']
    
        input_variables.loc[index,'player']=player
        input_variables.loc[index,'inning']=inning
        input_variables.loc[index,'against']=against
        input_variables.loc[index,'home']=home
        input_variables.loc[index,'toss_winner']=toss_winner
        index=index+1
        player = flask.request.form['player5']
    
        input_variables.loc[index,'player']=player
        input_variables.loc[index,'inning']=inning
        input_variables.loc[index,'against']=against
        input_variables.loc[index,'home']=home
        input_variables.loc[index,'toss_winner']=toss_winner
        index=index+1
        player = flask.request.form['player6']
    
        input_variables.loc[index,'player']=player
        input_variables.loc[index,'inning']=inning
        input_variables.loc[index,'against']=against
        input_variables.loc[index,'home']=home
        input_variables.loc[index,'toss_winner']=toss_winner
        index=index+1
        player = flask.request.form['player7']
    
        input_variables.loc[index,'player']=player
        input_variables.loc[index,'inning']=inning
        input_variables.loc[index,'against']=against
        input_variables.loc[index,'home']=home
        input_variables.loc[index,'toss_winner']=toss_winner
        index=index+1
        player = flask.request.form['player8']
    
        input_variables.loc[index,'player']=player
        input_variables.loc[index,'inning']=inning
        input_variables.loc[index,'against']=against
        input_variables.loc[index,'home']=home
        input_variables.loc[index,'toss_winner']=toss_winner
        index=index+1
        player = flask.request.form['player9']
    
        input_variables.loc[index,'player']=player
        input_variables.loc[index,'inning']=inning
        input_variables.loc[index,'against']=against
        input_variables.loc[index,'home']=home
        input_variables.loc[index,'toss_winner']=toss_winner
        index=index+1
        player = flask.request.form['player10']
    
        input_variables.loc[index,'player']=player
        input_variables.loc[index,'inning']=inning
        input_variables.loc[index,'against']=against
        input_variables.loc[index,'home']=home
        input_variables.loc[index,'toss_winner']=toss_winner
        index=index+1
        player = flask.request.form['player11']
    
        input_variables.loc[index,'player']=player
        input_variables.loc[index,'inning']=inning
        input_variables.loc[index,'against']=against
        input_variables.loc[index,'home']=home
        input_variables.loc[index,'toss_winner']=toss_winner
        index=index+1


        
        against = flask.request.form['team1']
        home=0
        selection = flask.request.form['inning']
        toss=flask.request.form['toss']
        if toss=='team2':
            toss_winner=1
            if selection=='batting':
                inning=1
            else :
                inning=2
        else :
            toss_winner=0
            if selection=='batting':
                inning=2
            else :
                inning=1
        
        player = flask.request.form['player12']
    
        input_variables.loc[index,'player']=player
        input_variables.loc[index,'inning']=inning
        input_variables.loc[index,'against']=against
        input_variables.loc[index,'home']=home
        input_variables.loc[index,'toss_winner']=toss_winner
        index=index+1
    
        player = flask.request.form['player13']
    
        input_variables.loc[index,'player']=player
        input_variables.loc[index,'inning']=inning
        input_variables.loc[index,'against']=against
        input_variables.loc[index,'home']=home
        input_variables.loc[index,'toss_winner']=toss_winner
        index=index+1
        player = flask.request.form['player14']
    
        input_variables.loc[index,'player']=player
        input_variables.loc[index,'inning']=inning
        input_variables.loc[index,'against']=against
        input_variables.loc[index,'home']=home
        input_variables.loc[index,'toss_winner']=toss_winner
        index=index+1
        player = flask.request.form['player15']
    
        input_variables.loc[index,'player']=player
        input_variables.loc[index,'inning']=inning
        input_variables.loc[index,'against']=against
        input_variables.loc[index,'home']=home
        input_variables.loc[index,'toss_winner']=toss_winner
        index=index+1
        player = flask.request.form['player16']
    
        input_variables.loc[index,'player']=player
        input_variables.loc[index,'inning']=inning
        input_variables.loc[index,'against']=against
        input_variables.loc[index,'home']=home
        input_variables.loc[index,'toss_winner']=toss_winner
        index=index+1
        player = flask.request.form['player17']
    
        input_variables.loc[index,'player']=player
        input_variables.loc[index,'inning']=inning
        input_variables.loc[index,'against']=against
        input_variables.loc[index,'home']=home
        input_variables.loc[index,'toss_winner']=toss_winner
        index=index+1
        player = flask.request.form['player18']
    
        input_variables.loc[index,'player']=player
        input_variables.loc[index,'inning']=inning
        input_variables.loc[index,'against']=against
        input_variables.loc[index,'home']=home
        input_variables.loc[index,'toss_winner']=toss_winner
        index=index+1
        player = flask.request.form['player19']
    
        input_variables.loc[index,'player']=player
        input_variables.loc[index,'inning']=inning
        input_variables.loc[index,'against']=against
        input_variables.loc[index,'home']=home
        input_variables.loc[index,'toss_winner']=toss_winner
        index=index+1
        player = flask.request.form['player20']
    
        input_variables.loc[index,'player']=player
        input_variables.loc[index,'inning']=inning
        input_variables.loc[index,'against']=against
        input_variables.loc[index,'home']=home
        input_variables.loc[index,'toss_winner']=toss_winner
        index=index+1
        player = flask.request.form['player21']
    
        input_variables.loc[index,'player']=player
        input_variables.loc[index,'inning']=inning
        input_variables.loc[index,'against']=against
        input_variables.loc[index,'home']=home
        input_variables.loc[index,'toss_winner']=toss_winner
        index=index+1
        player = flask.request.form['player22']
    
        input_variables.loc[index,'player']=player
        input_variables.loc[index,'inning']=inning
        input_variables.loc[index,'against']=against
        input_variables.loc[index,'home']=home
        input_variables.loc[index,'toss_winner']=toss_winner
        index=index+1
        
        
        

        
        
        
        
        
        
        
        
        
        input_variables.replace(['MS Dhoni','AT Rayudu','KM Asif','DL Chahar','DJ Bravo','F du Plessis','Imran Tahir','Narayan Jagadeesan','KV Sharma','KM Jadhav','L Ngidi','MJ Santner','Monu Singh','M Vijay','RA Jadeja','Ruturaj Gaikwad','SR Watson','SN Thakur','J Hazlewood','PP Chawla','R Sai Kishore','SM Curran','SS Iyer','AM Rahane','A Mishra','Avesh Khan','AR Patel','HV Patel','I Sharma','K Rabada','K Paul','PP Shaw','R Ashwin','RR Pant','S Lamichhane','S Dhawan','A Carey','L Yadav','MP Stoinis','MM Sharma','SO Hetmyer','T Deshpande','D Sams','A Nortje','V Kohli','AB de Villiers','D Padikkal','Gurkeerat Singh','MM Ali','Mohammed Siraj','N Saini','PA Patel','P Negi','S Dube','UT Yadav','Washington Sundar','YS Chahal','AJ Finch','CH Morris','DW Steyn','I Udana','J Philippe','P Deshpande','S Ahamad','A Zampa','RG Sharma','AP Tare','A Singh','AS Roy','DS Kulkarni','HH Pandya','Ishan Kishan','JJ Bumrah','J Yadav','KA Pollard','KH Pandya','MJ McClenaghan','Q de Kock','RD Chahar','S Rutherford','SA Yadav','TA Boult','CA Lynn','D Deshmukh','M Khan','NM Coulter-Nile','Prince Balwant Rai Singh','SS Tiwary','J Pattinson','KL Rahul','Arshdeep Singh','CH Gayle','D Nalkande','K Gowtham','GC Viljoen','Harpreet Brar','J Suchith','KK Nair','Mandeep Singh','MA Agarwal','Mohammed Shami','Mujeeb Ur Rahman','M Ashwin','N Pooran','SN Khan','CJ Jordan','DJ Hooda','GJ Maxwell','Ishan Porel','JDS Neesham','Prabhsimran Singh','R Bishnoi','S Cottrell','T Dhillon','KD Karthik','AD Russell','Kamlesh Nagarkoti','Kuldeep Yadav','LH Ferguson','N Rana','M Prasidh Krishna','RK Singh','S Sandeep Warrier','Shivam Mavi','Shubman Gill','SD Lad','SP Narine','Chris Green','EJG Morgan','M Siddharth','NS Naik','PJ Cummins','Ra Tripathi','Tom Banton','Varun Chakravarthy','SPD Smith','AS Rajpoot','BA Stokes','JC Archer','JC Buttler','MK Lomror','M Vohra','M Markande','R Tewatia','R Parag','SV Samson','Shashank Singh','S Gopal','VR Aaron','Akash Singh','Anirudha Joshi','Anuj Rawat','AJ Tye','DA Miller','JD Unadkat','Kartik Tyagi','O Thomas','RV Uthappa','TK Curran','Yashasvi Jaiswal','DA Warner','Abhishek Sharma','Basil Thampi','B Kumar','B Stanlake','JM Bairstow','KS Williamson','MK Pandey','Mohammad Nabi','Rashid Khan','Sandeep Sharma','S Nadeem','SP Goswami','S Kaul','Khaleel Ahmed','T Natarajan','V Shankar','WP Saha','Abdul Samad','Fabian Allen','MR Marsh','Priyam Garg','Sandeep Bavanaka','Sanjay Yadav','Virat Singh'],['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60','61','62','63','64','65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80','81','82','83','84','85','86','87','88','89','90','91','92','93','94','95','96','97','98','99','100','101','102','103','104','105','106','107','108','109','110','111','112','113','114','115','116','117','118','119','120','121','122','123','124','125','126','127','128','129','130','131','132','133','134','135','136','137','138','139','140','141','142','143','144','145','146','147','148','149','150','151','152','153','154','155','156','157','158','159','160','161','162','163','164','165','166','167','168','169','170','171','172','173','174','175','176','177','178','179','180','181','182','183','184','185'],inplace=True)
        input_variables.replace(['Royal Challengers Bangalore', 'Sunrisers Hyderabad','Rising Pune Supergiant', 'Mumbai Indians','Kolkata Knight Riders', 'Gujarat Lions', 'Kings XI Punjab','Delhi Daredevils', 'Chennai Super Kings', 'Rajasthan Royals','Kochi Tuskers Kerala'],[ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11],inplace=True)
        lm_pred = lm.predict(input_variables)
        rf_pred=rf.predict(input_variables)
        
        
        input_variables['lm']=lm_pred
        input_variables['rf']=rf_pred
        
        
        
        input_variables['player'].replace(['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60','61','62','63','64','65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80','81','82','83','84','85','86','87','88','89','90','91','92','93','94','95','96','97','98','99','100','101','102','103','104','105','106','107','108','109','110','111','112','113','114','115','116','117','118','119','120','121','122','123','124','125','126','127','128','129','130','131','132','133','134','135','136','137','138','139','140','141','142','143','144','145','146','147','148','149','150','151','152','153','154','155','156','157','158','159','160','161','162','163','164','165','166','167','168','169','170','171','172','173','174','175','176','177','178','179','180','181','182','183','184','185'],['MS Dhoni','AT Rayudu','KM Asif','DL Chahar','DJ Bravo','F du Plessis','Imran Tahir','Narayan Jagadeesan','KV Sharma','KM Jadhav','L Ngidi','MJ Santner','Monu Singh','M Vijay','RA Jadeja','Ruturaj Gaikwad','SR Watson','SN Thakur','J Hazlewood','PP Chawla','R Sai Kishore','SM Curran','SS Iyer','AM Rahane','A Mishra','Avesh Khan','AR Patel','HV Patel','I Sharma','K Rabada','K Paul','PP Shaw','R Ashwin','RR Pant','S Lamichhane','S Dhawan','A Carey','L Yadav','MP Stoinis','MM Sharma','SO Hetmyer','T Deshpande','D Sams','A Nortje','V Kohli','AB de Villiers','D Padikkal','Gurkeerat Singh','MM Ali','Mohammed Siraj','N Saini','PA Patel','P Negi','S Dube','UT Yadav','Washington Sundar','YS Chahal','AJ Finch','CH Morris','DW Steyn','I Udana','J Philippe','P Deshpande','S Ahamad','A Zampa','RG Sharma','AP Tare','A Singh','AS Roy','DS Kulkarni','HH Pandya','Ishan Kishan','JJ Bumrah','J Yadav','KA Pollard','KH Pandya','MJ McClenaghan','Q de Kock','RD Chahar','S Rutherford','SA Yadav','TA Boult','CA Lynn','D Deshmukh','M Khan','NM Coulter-Nile','Prince Balwant Rai Singh','SS Tiwary','J Pattinson','KL Rahul','Arshdeep Singh','CH Gayle','D Nalkande','K Gowtham','GC Viljoen','Harpreet Brar','J Suchith','KK Nair','Mandeep Singh','MA Agarwal','Mohammed Shami','Mujeeb Ur Rahman','M Ashwin','N Pooran','SN Khan','CJ Jordan','DJ Hooda','GJ Maxwell','Ishan Porel','JDS Neesham','Prabhsimran Singh','R Bishnoi','S Cottrell','T Dhillon','KD Karthik','AD Russell','Kamlesh Nagarkoti','Kuldeep Yadav','LH Ferguson','N Rana','M Prasidh Krishna','RK Singh','S Sandeep Warrier','Shivam Mavi','Shubman Gill','SD Lad','SP Narine','Chris Green','EJG Morgan','M Siddharth','NS Naik','PJ Cummins','Ra Tripathi','Tom Banton','Varun Chakravarthy','SPD Smith','AS Rajpoot','BA Stokes','JC Archer','JC Buttler','MK Lomror','M Vohra','M Markande','R Tewatia','R Parag','SV Samson','Shashank Singh','S Gopal','VR Aaron','Akash Singh','Anirudha Joshi','Anuj Rawat','AJ Tye','DA Miller','JD Unadkat','Kartik Tyagi','O Thomas','RV Uthappa','TK Curran','Yashasvi Jaiswal','DA Warner','Abhishek Sharma','Basil Thampi','B Kumar','B Stanlake','JM Bairstow','KS Williamson','MK Pandey','Mohammad Nabi','Rashid Khan','Sandeep Sharma','S Nadeem','SP Goswami','S Kaul','Khaleel Ahmed','T Natarajan','V Shankar','WP Saha','Abdul Samad','Fabian Allen','MR Marsh','Priyam Garg','Sandeep Bavanaka','Sanjay Yadav','Virat Singh'],inplace=True)
        input_variables['against'].replace([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11],['Royal Challengers Bangalore', 'Sunrisers Hyderabad','Rising Pune Supergiant', 'Mumbai Indians','Kolkata Knight Riders', 'Gujarat Lions', 'Kings XI Punjab','Delhi Daredevils', 'Chennai Super Kings', 'Rajasthan Royals','Kochi Tuskers Kerala'],inplace=True)

        
        
        input_variables_lm=input_variables.sort_values(by=['lm'],ascending=False)
        input_variables_lm.reset_index(drop=True,inplace=True)
        input_variables_rf=input_variables.sort_values(by=['rf'],ascending=False)
        input_variables_rf.reset_index(drop=True,inplace=True)
        
        return flask.render_template('main.html',
                                     
                                     tables=[input_variables_lm.to_html(classes='data')], titles=input_variables_lm.columns.values,
                                     tables1=[input_variables_rf.to_html(classes='data')])
if __name__ == '__main__':
    app.run()