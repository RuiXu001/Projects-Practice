# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 15:43:26 2022

@author: xurui
"""
# refer to https://www.youtube.com/watch?v=eN_3d4JrL_w

#https://www.indeed.com/jobs?q=Data%20Analyst&l=San%20Jose%2C%20CA&explvl=entry_level&vjk=1ba83f797065a8c6
#https://www.indeed.com/jobs?q=Data%20Analyst&l=San%20Jose%2C%20CA&explvl=entry_level&fromage=3&vjk=af6c50ad9753fa9b
from bs4 import BeautifulSoup
import urllib
import re
import collections
from time import sleep
import datetime
import pandas as pd
import os
#%%
def get_info(url):
    '''
    

    Parameters
    ----------
    url : 
        url of one indeed search page

    Returns
    -------
    dftmp : df
        df of job records of one indeed page.

    '''
    try:
        html = urllib.request.urlopen(url).read()
        print('Reading ', url)
    except:
        print('invalid url!')
        #return None
    soup = BeautifulSoup(html, "lxml")
    count_page=soup.find('div',{'id':'searchCountPages'}).text
    print(count_page)
    #cards = soup.find_all('div',{'class':'slider_container'})
    links=soup.find_all('a',{'data-hiring-event':'false'})
    dftmp=pd.DataFrame(columns=['labels','date','jobtitle','companyname','location',
                               'salary','jobdescription','link','others'])
    for l in links: # go to each view job page and get the info.
        link=base_url+l['href']
        html = urllib.request.urlopen(link).read()
        soup = BeautifulSoup(html, "lxml")
        label=query
        title=soup.h1.text
        #info=soup.find('div',{'class':'jobsearch-JobTab-content'}).find_all('div')
        c_info=soup.find('div',{'class':'jobsearch-CompanyInfoContainer'})#.find_all('div')
        info=[]
        for i in c_info.div.find_all('div',{'class':''}):
            info.append(i.text)
        company=info[1]#c_info.div.text # company and location
        location=info[3]#c_info.find_all('div')[-2].text #this may not work...
        #company=company.replace(location,'')
        try:
            salary=soup.find('div',{'class':'jobsearch-JobMetadataHeader-item'}).text
        except:
            salary=''
        post=soup.find('div',{'class':'jobsearch-JobMetadataFooter'}).text
        if 'Today' in post or 'Just' in post:
            date=today 
        else:
            date=today-datetime.timedelta(days=int(re.findall('[0-9]+', post)[0]))
        date=date.strftime('%Y-%m-%d')
        jd= soup.find('div',{'id':'jobDescriptionText'}).text
        add={'labels':label,'date':date,'jobtitle':title,'companyname':company,
             'jobdescription':jd,'salary':salary,'location':location,'link':link}
        dftmp=dftmp.append(add, ignore_index=True)
        print('Adding one record: ', add['jobtitle'],' ', add['companyname'], ' ',add['location'])
        sleep(5)#sleep
    return dftmp

def next_page(url):
    '''

    Parameters
    ----------
    url : 
        url of one indeed search page.

    Returns
    -------
    next_url : 
        url of the next search page.

    '''
    html = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(html, "lxml")
    try: # one page 
        navigation=soup.find_all('nav', {'role':'navigation'})[-1]
        next_=navigation.find('a',{'aria-label':'Next'}) #next page
        if not next_:
            print('End of the list. No next page.')
            return None
        else:
            next_url='https://www.indeed.com'+next_['href']
            return next_url
    except:
        return None
#%%

base_url = 'https://www.indeed.com'
today = datetime.datetime.today()

if not os.path.exists('D:/resume/jobs.csv'):
    df = pd.DataFrame(columns=['labels','date','jobtitle','companyname','location',
                               'salary','jobdescription','link','others'])
    df.to_csv('D:/resume/jobs.csv',index=False)
    print('created job.csv')
else:
    df=pd.read_csv('D:/resume/jobs.csv')
    print(f'open job.csv\nthere are {len(df)} records')

#query='Data%20Analyst'
#query='Data%20Scientist'
query='Business%20Analyst'
city, state='San%20Jose', 'CA'
#url = base_url + '/jobs?q=' + query + '&l=' + city + '%2C+' + state + '&explvl=entry_level&fromage=14&sort=date'
url = base_url + '/jobs?q=' + query + '&l=' + city + '%2C+' + state + '&explvl=entry_level&fromage=3&sort=date'


while url: # while the give url or 
    dftmp = get_info(url)
    df=pd.concat([df,dftmp],ignore_index=True)
    df.drop_duplicates(subset=['jobtitle', 'companyname'],inplace=True)
    df=df.reset_index(drop=True)
    #df.drop(columns=['index'], inplace=True)
    print(f'{len(df)} records')
    url=next_page(url)

df.sort_values('date', ascending=False, ignore_index=True, inplace=True)
df.to_csv('D:/resume/jobs.csv',index=False)
print(f'File saved. There are {len(df)} records.')





