import random,logging,time,sys,os,datetime

class Logger(object):    
    def __init__(self):
        self.logger,self.log_filename = self.getLogger()

    def getLogger(self):        
        random_str = str(random.randint(1,10000))
    
        now = int(time.time()) 
        timeArray = time.localtime(now)
        timeStamp = time.strftime("%Y%m%d%H%M%S", timeArray)
        log_path = "log/acc" +time.strftime("%Y%m%d", timeArray)
        log_filename = log_path+'/'+timeStamp+"_"+ random_str+'.log'
        
        program = os.path.basename(sys.argv[0])
        logger = logging.getLogger(program) 
        
        if not os.path.exists("log"):
            os.mkdir("log")
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
                            datefmt='%a, %d %b %Y %H:%M:%S',
                            filename=log_filename,
                            filemode='w')
        logging.root.setLevel(level=logging.INFO)
        logger.info("running %s" % ' '.join(sys.argv)) 
        logger.info(log_path+'/'+timeStamp+"_"+ random_str+'.log')
 
        return logger,log_filename
    
    def info(self,text):
#        time_str = datetime.datetime.now().isoformat()
#        string="{}:  {}".format(time_str, text)
#            logger.info("{}: step {}, loss {:g}, acc {:g} ,positive {:g},negative {:g}".format(time_str, step, loss, accuracy,np.mean(score12),np.mean(score13)))
        self.logger.info(str(text))
    def getCSVLogger(self):
        from keras.callbacks import CSVLogger
        csv_logger = CSVLogger(self.log_filename, append=True, separator=';')
        return csv_logger

if __name__ == "__main__":
    
    logger = Logger()
    logger.info("sb")



    