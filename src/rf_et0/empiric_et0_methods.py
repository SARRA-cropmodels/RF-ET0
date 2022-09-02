import numpy as np
from matplotlib import pyplot as plt

def calc_empiric_ET0_estimates(df_weather):

    ### MODIF
    # on divise HR par 100 pour le ramener dans [0,1]
    df_weather["es-e"]=df_weather["VAP"]-(df_weather["HR"]/100*df_weather["VAP"])
    df_weather["delta"]=(np.exp(20.386 - (5132 / (df_weather["TEMP"] + 273))))/7.501 #on divise par 7.501 pour passer de mmHg à kPa
    lambda2=2.2647


    # AERODYNAMIC
    df_weather["ET0-dalton"]=(0.07223*df_weather["WIND"]+0.3648)*df_weather["es-e"] #OK
    df_weather["ET0-trabert"]=0.3075*np.sqrt(df_weather["WIND"])*df_weather["es-e"] #OK
    df_weather["ET0-penman"]=0.35*(1+0.24*df_weather["WIND"])*df_weather["es-e"] #OK
    df_weather["ET0-rohwer"]=0.44*(1+0.27*df_weather["WIND"])*df_weather["es-e"] #OK
    df_weather["ET0-mahringer"]=0.15072*np.sqrt(3.6)*df_weather["WIND"]*df_weather["es-e"] #OK

    # TEMPERATURE
    df_weather["ET0-hargreaves"]=0.0135*0.408*df_weather["IRRAD"]/1000000*(df_weather["TEMP"]+17.8) #OK

    # RADIATION
    df_weather["ET0-jensen-haise"]=0.025*(df_weather["TEMP"]-3)*df_weather["IRRAD"]/1000000 #OK
    df_weather["ET0-abtew"]=0.53*df_weather["IRRAD"]/1000000/lambda2 #OK
    df_weather["ET0-oudin"]=df_weather["IRRAD"]/1000000*((df_weather["TEMP"]+5)/100) #OK
    df_weather["ET0-irmak-a"]=0.149*df_weather["IRRAD"]/1000000+0.079*df_weather["TEMP"]-0.611
    df_weather["ET0-irmak-b"]=0.174*df_weather["IRRAD"]/1000000+0.00353*df_weather["TEMP"]-0.642
    df_weather["ET0-irmak-c"]=0.156*df_weather["IRRAD"]/1000000-0.0112*df_weather["TMAX"]+0.0733*df_weather["TMIN"]-0.478

    # COMBINATORY
    df_weather["ET0-valiantzas-a"]=0.0393*df_weather["IRRAD"]/1000000 *np.sqrt(df_weather["TEMP"]+9.5)-0.19*(df_weather["IRRAD"]/1000000)**0.6*(np.abs(df_weather["LAT"])*3.14/180)**0.15+0.048*(df_weather["TEMP"]+20)*(1-df_weather["HR"]/100)*df_weather["WIND"]**0.7 #OK
    df_weather["ET0-valiantzas-b"]=0.0393*df_weather["IRRAD"]/1000000 *np.sqrt(df_weather["TEMP"]+9.5)-0.19*(df_weather["IRRAD"]/1000000)**0.6*(np.abs(df_weather["LAT"])*3.14/180)**0.15+0.078*(df_weather["TEMP"]+20)*(1-df_weather["HR"]/100) #OK
    df_weather["ET0-valiantzas-c"]=0.0393*df_weather["IRRAD"]/1000000 *np.sqrt(df_weather["TEMP"]+9.5)-0.19*(df_weather["IRRAD"]/1000000)**0.6*(np.abs(df_weather["LAT"])*3.14/180)**0.15+0.0061*(df_weather["TEMP"]+20)*(1.12*df_weather["TEMP"]-df_weather["TMIN"]-2)**0.7 #OK

    return df_weather


def plot_empiric_ET0_estimates(df_weather):
    plt.figure(num=None, figsize=(16, 8), dpi=150, facecolor='w', edgecolor='k')


    # AERODYNAMIC
    plt.plot(df_weather["Jour"],df_weather["ET0-dalton"],label="ET0-dalton",linewidth=.7)
    plt.plot(df_weather["Jour"],df_weather["ET0-trabert"],label="ET0-trabert",linewidth=.7)
    plt.plot(df_weather["Jour"],df_weather["ET0-penman"],label="ET0-penman",linewidth=.7)
    plt.plot(df_weather["Jour"],df_weather["ET0-rohwer"],label="ET0-rohwer",linewidth=.7)
    plt.plot(df_weather["Jour"],df_weather["ET0-mahringer"],label="ET0-mahringer",linewidth=.7)

    # TEMPERATURE
    plt.plot(df_weather["Jour"],df_weather["ET0-hargreaves"],label="ET0-hargreaves",linewidth=.7)

    # RADIATION
    plt.plot(df_weather["Jour"],df_weather["ET0-jensen-haise"],label="ET0-jensen-haise",linewidth=.7)
    plt.plot(df_weather["Jour"],df_weather["ET0-abtew"],label="ET0-abtew",linewidth=.7)
    plt.plot(df_weather["Jour"],df_weather["ET0-oudin"],label="ET0-oudin",linewidth=.7)
    plt.plot(df_weather["Jour"],df_weather["ET0-irmak-a"],label="ET0-irmak-a",linewidth=.7)
    plt.plot(df_weather["Jour"],df_weather["ET0-irmak-b"],label="ET0-irmak-b",linewidth=.7)
    plt.plot(df_weather["Jour"],df_weather["ET0-irmak-c"],label="ET0-irmak-c",linewidth=.7)

    # COMBINATORY
    plt.plot(df_weather["Jour"],df_weather["ET0-valiantzas-a"],label="ET0-valiantzas-a",linewidth=.7)
    plt.plot(df_weather["Jour"],df_weather["ET0-valiantzas-b"],label="ET0-valiantzas-b",linewidth=.7)
    plt.plot(df_weather["Jour"],df_weather["ET0-valiantzas-c"],label="ET0-valiantzas-c",linewidth=.7)


    # REFERENCE
    plt.plot(df_weather["Jour"],df_weather["ET0-PM"],label="ET0-PM",linewidth=2, color="blue")

    # RESULT
    plt.plot(df_weather["Jour"],df_weather["ET0-RF"],label="ET0-RF",linewidth=2, color="red")

    plt.xticks(rotation=70)
    plt.title("ET0 (mm) empiric model prediction vs ET0-PM")
    plt.legend() 
    plt.show()