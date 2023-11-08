import numpy as np

def get_mean_std_hbec(name, data_hbec) :
    
    """ 
    Get the vectors of mean data and standard deviation data 
    
    name : name of the species (Per, Cry, Rev, Ror, Bmal, Clock)
    data_hbec : python dictionary with the HBEC data
    
    """
    
    array_paralog = sum_paralog_hbec(name, data_hbec)
    
    mean_vector = np.array([np.nanmean(l) for l in array_paralog])  
    std_vector = np.array([np.nanstd(l) for l in array_paralog]) 
    
    return mean_vector*10**3, std_vector*10**3



def sum_paralog_hbec(name, data_hbec) :
    
    """ 
    Sum the data of each paralog into a numpy array
    
    name : name of the species (Per, Cry, Rev, Ror, Bmal, Clock)
    data_hbec : python dictionary with the HBEC data
    
    """
    
    if name == "PER" :
        per2_hbec = data_hbec["ctrl"]["PER2"]["pts"]

        array_paralog = np.copy(per2_hbec)
        for ct in range(len(data_hbec["CTs"])) :
            for rep in range(3) :
                array_paralog[ct, rep] = np.nansum([per2_hbec[ct,rep]])

    if name == "CRY" :
        cry1_hbec = data_hbec["ctrl"]["CRY1"]["pts"]
        cry2_hbec = data_hbec["ctrl"]["CRY2"]["pts"]

        array_paralog = np.copy(cry1_hbec)
        for ct in range(len(data_hbec["CTs"])) :
            for rep in range(3) :
                array_paralog[ct, rep] = np.nansum([cry1_hbec[ct,rep], cry2_hbec[ct,rep]])

    if name == "REV-ERB" :
        nr1d1_hbec = data_hbec["ctrl"]["NR1D1"]["pts"]
        nr1d2_hbec = data_hbec["ctrl"]["NR1D2"]["pts"]

        array_paralog = np.copy(nr1d1_hbec)
        for ct in range(len(data_hbec["CTs"])) :
            for rep in range(3) :
                array_paralog[ct, rep] = np.nansum([nr1d1_hbec[ct,rep], nr1d2_hbec[ct,rep]])

    if name == "ROR" :
        rora_hbec = data_hbec["ctrl"]["RORA"]["pts"]
        rorb_hbec = data_hbec["ctrl"]["RORB"]["pts"]
        rorc_hbec = data_hbec["ctrl"]["RORC"]["pts"]

        array_paralog = np.copy(rora_hbec)
        for ct in range(len(data_hbec["CTs"])) :
            for rep in range(3) :
                array_paralog[ct, rep] = np.nansum([rora_hbec[ct,rep], rorb_hbec[ct,rep], rorc_hbec[ct,rep]])

    if name == "BMAL1" :
        array_paralog = np.copy(data_hbec["ctrl"]["ARNTL"]["pts"])
        
    if name == "CLOCK" :
        array_paralog = data_hbec["ctrl"]["CLOCK"]["pts"]

    return array_paralog