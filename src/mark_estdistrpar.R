#' Estimation based on probability generating functions
estmusigPGF <- function(dat, js=1:nrow(dat), allefs, whnew=NULL, ngrid=25, optimizer="rsolnp", startmat=NULL, prior=TRUE, 
                        priormu = c(1,(1/6)^2),priordisp = c(0.5,1/6), cont=NULL,distr ="ln",modus="samedisp"){
  if(is.null(cont)) cont <- list(outer.iter=10)
  nT <- nrow(allefs)
  nsam <- ncol(dat)
  if(is.null(whnew)){print("Assuming all samples are from the same source, so share parameters"); modus <- "samedisp";whnew<-rep(FALSE,nsam)}
  
  #if(distr =="trnormal") {pdist <- function(x,f,mu,disp){trun0 <- pnorm(0,f*mu, sd=f*disp);return((pnorm(x,f*mu, sd=f*disp)-trun0)/(1-trun0))}; lb <- 0.01; ub <- 10^10}
  if(distr =="normal") {pdist <- function(x,f,mu,disp){return(pnorm(x,f*mu, sd=f*disp))}; lb <- 0.01; ub <- 10^10}
  if(distr =="ln") {pdist <- function(x,f,mu,disp){return(plnorm(x,log(f) + mu, sd=disp))}; lb <- 0.05; ub <- 20}
  if(distr =="nb") {pdist <- function(x,f,mu,disp){return(pnbinom(x/f,mu=exp(mu),size=disp))}; lb <- 0.0001; ub <- 100} #because mu is searched on the log-scale

  if(modus=="samedisp") ndispT <- nT
  if(modus=="scaledisp") ndispT <- nT+1
  if(modus=="difdisp") ndispT <- 2*(nT-1)+1
  
  params2mudisp <- function(params,modus){
   if(modus=="samedisp") return( c(params,params[-(1:(nT+1))]))
   if(modus=="scaledisp") return(c(params[1:nT],params[nT+1], params[(nT+2):(2*nT)], params[(2*nT +1)] * params[(nT+2):(2*nT)]))
   if(modus=="difdisp") return(params)
  } 
  
  logpenalty <- function(mudisps, priormu,priordisp,nT,distr){
    mut <- mudisps[1:nT]
    dispt <- mudisps[-(1:nT)]
    if(distr=="ln"){
    lp <- sum(dnorm(mut,priormu[1],sd=sqrt(1/priormu[2]),log=T)) +
      sum(dgamma(dispt,priordisp[1],rate=priordisp[2],log=T)) 
    } else {
      lp <- sum(dnorm(mut,priormu[1],sd=sqrt(1/priormu[2]),log=T)) +
      sum(dlnorm(dispt,mean=0,sd=1,log=T)) 
    }
    return(lp)
  }
  
  resj <- function(j){
    #j<-1
    #initialization for mus and disps when startmat is NULL
    param <- c(rep(2, nT),rep(1,ndispT))
    datj <- dat[j,]
    #j<-1
    whj <- which(js==j)
    print(whj)
    Ydens.loop <- function(i, allefs, mudisps, distr){
      #j=1;efs <- rep(1/nT, nT) ;i<-6;mudisps <- param
      testi <- whnew[i] 
      muTa <- mudisps[1:nT];
      if(!testi) dispa <- mudisps[(nT+1):(2*nT)] else dispa <- c(mudisps[nT+1],mudisps[(2*nT+1):(3*nT-1)])
      yi <- as.numeric(datj[i])
      #print(yi)
      
      efsi <- allefs[,i]
      
      whnz <- which(efsi >= 0.001) #
      #lf <- log(efsi[whnz])
      ef <- efsi[whnz]
      muT <- muTa[whnz]
      dispT <- dispa[whnz]
      nTnz <- length(whnz)
      
      if(yi==0) {a <- 0.5; step <- 1; ngridi <- 0} else {
        if(distr=="nb") { ngridi <- min(ngrid,yi)} else ngridi <- ngrid
        step <- yi/ngridi
        a<- rev(c(0,1:ngridi)+0.5)*step
      }
      
      #t=1
      t<-1
      #convt <- plnorm(a,lf[t] + muT[t],disp[t]) - plnorm(a-step,muT[t]+lf[t],disp[t]); 
      convt <- pdist(a,ef[t],muT[t],dispT[t]) - pdist(a-step,ef[t],muT[t],dispT[t]) 
      if(nTnz >1){
        for(t in 2:nTnz){
          #t<-2
          at <- pdist(a,ef[t],muT[t],dispT[t]) - pdist(a-step,ef[t],muT[t],dispT[t]) ; 
          convo <- conv(convt,at) #sorted from high to low powers
          convt <- rev(rev(convo)[1:(ngridi+1)]) #sorted from high to low powers; only powers yi ... 0 are needed
        }
      }
      intpgf<-convt[1]/step
      return(max(10^(-100),intpgf)) 
    }
    

    loglik.loop <- function(params){
      #mudisps <- initpar
      mudisps <- params2mudisp(params,modus)
      
      if(!prior) return(-sum(log(sapply(1:nsam, Ydens.loop, mudisps=mudisps, allefs=allefs,distr=distr)))) else {
      minusloglik <- -sum(log(sapply(1:nsam, Ydens.loop, mudisps=mudisps, allefs=allefs,distr=distr)))
      logpenal <- -logpenalty(mudisps,priormu=priormu,priordisp=priordisp,nT=nT,distr=distr)
      return(minusloglik+logpenal) 
      }
    }
    # loglik.loop(param)
    # loglik.loop(truepars[j,])
    #pmt <- proc.time()
    
    # if(is.null(startmat)) {initpar <- param} else 
    #   {
    #   if(is.na(startmat[j,1])) {initpar <- param} else 
    #     {
    #     initpar <- startmat[j,]
    #     }
    #   }
    
    initpar <- param
    if(!is.null(startmat)) 
    {
    whnotNA <- which(!is.na(startmat[j,]))
    if(length(whnotNA)>0) initpar[whnotNA] <- startmat[j,whnotNA] 
    }
    
    print("Starting values:")
    print(initpar) 
    loglik.loop(initpar)
      
    if(optimizer=="rsolnp") optres <- try(solnp(par = initpar, loglik.loop,  
                                                LB = c(rep(-10,nT),rep(lb,ndispT)), UB=c(rep(20,nT),rep(ub,ndispT)),
                                                control=cont )) else {
                                                  optres <- try(optim(par = initpar, loglik.loop, lower =  c(rep(-10,nT),rep(lb,ndispT)), 
                                                                      upper=c(rep(20,nT),rep(ub,ndispT)), method = "L-BFGS-B"))  
                                                  print(optres$value)
                                                  print(optres$par)
                                                }
    if(distr=="normal"){datj <- exp(datj)-1;distrib = "ln"} else distrib <- distr
    loglik <- function(params){
      mudisps <- params2mudisp(params,modus)
      return(-sum(log(sapply(1:nsam, Ydens.loop, mudisps=mudisps, allefs=allefs,distr=distrib)))) 
    }
    
    loglikopt <- loglik(optres$pars)
    print(loglikopt)
    optres <- c(optres,likopt=list(loglikopt))
    return(optres)
  } # end function i
  #is <- 1:2
  resall <- lapply(js,resj)
  return(resall)
}



#estimates from single tissue
estmusigsingle <- function(datsingle, fssingle, limma=T, distr="ln"){
  #dat<- datandpars$datsingle; fssingle= datandpars$fssingle
  nT <- nrow(fssingle)
  nsingle <- ncol(fssingle)
  
  if(distr =="ln"){
  allmus <- alldisps <- c()
  
  for(t in 1:nT){
    #t<-1
    whin <- which(fssingle[t,]==1)
   dat <- log(datsingle[,whin])
   mus <- apply(dat,1,mean)
   allmus <- rbind(allmus,mus)
   if(limma){
     design <- cbind(Intercept=rep(1,nsingle/nT))
     fit <- lmFit(dat,design)
     fit <- eBayes(fit)
     varsshrink <- fit$s2.post
     disps <- 1/varsshrink
   } else disps <- 1/apply(dat,1,var)
   alldisps <- rbind(alldisps,disps)
  }
  allpars <- rbind(allmus,alldisps)
  rownames(allpars) <- c(paste("mu",1:nT,sep=""),paste("disp",1:nT,sep=""))
  } #end ln
  if(distr =="nb"){
   datsingle <- round(datsingle,0) #data should be integers
    
   allmus <- alldisps <- c()
   for(t in 1:nT){
     #t<-1
     whin <- which(fssingle[t,]==1)
     dat <- datsingle[,whin]
     mus <- apply(dat,1,mean)
     allmus <- rbind(allmus,mus)
     d <- DGEList(counts = dat) #using edgeR
     d.CR <- estimateCommonDisp(d)
     if(nrow(dat) >= 100) d.CR <- estimateTrendedDisp(d.CR)
     d.CR <- estimateTagwiseDisp(d.CR)
     disps <- d.CR$tagwise.dispersion
     #var2 <- mus + mus^2*disps
     alldisps <- rbind(alldisps,disps)
   }
   allpars <- rbind(sapply(log(allmus),max,y=-5),alldisps)  #log to make it easier for optimization purposes
   rownames(allpars) <- c(paste("logmu",1:nT,sep=""),paste("disp",1:nT,sep=""))
  } #end nb
  return(t(allpars))
}

#init estimate for (mu,sigma) of the remainder component
parsremain <- function(datbulk,js,musingle,dispsingle,initf=NULL,ngrid=25, optimizer="rsolnp", prior=TRUE,
                            priormu = c(1,(1/6)^2),priordisp = c(0.5,1/6),startmat=NULL,cleverinit=TRUE, distr="ln"){

  #FIRST f SHOULD BE REMAINDER COMPONENT!!! 
  nsam <- ncol(datbulk)
  
  #assumes priormeans are known for nT cell types
  nT <- ncol(musingle)+1
  
  #initialize fs; uniform if no info
  if(is.null(initf)){
    allefs <- matrix(rep(rep(1/nT,nT),nsam),ncol=nsam)
  } else allefs <- initf
  
  if(distr =="ln") {pdist <- function(x,f,mu,disp){return(plnorm(x,log(f) + mu, sd=disp))}; lb <- 0.05; ub <- 20}
  if(distr =="nb") {pdist <- function(x,f,mu,disp){return(pnbinom(x/f,mu=exp(mu),size=disp))}; lb <- 0.0001; ub <- 100} #because mu is searched on the log-scale

  logpenalty <- function(j, mudispT, priormu,priordisp){
    mut <- mudispT[1]
    dispt <- mudispT[2]
    lp <- dnorm(mut,priormu[1],sd=sqrt(1/priormu[2]),log=T) +
      dgamma(dispt,priordisp[1],rate=priordisp[2],log=T) 
    return(lp)
  }
  
  resj <- function(j){
    #j<-1
    musj <- musingle[j,]
    dispsj <- dispsingle[j,]
    datj <- datbulk[j,]
    
    
    allconvo <- list()
    for(i in 1:nsam){
      #i<-1
      yi <- as.numeric(datj[i])
      efsi <- allefs[-1,i]
      
      whnz <- which(efsi >= 0.001) #
      
      ef <- efsi[whnz]
      muT <- musj[whnz]
      disp <- dispsj[whnz] 
      nTnz <- length(whnz)
      
      
      if(yi==0) {a <- 0.5; step <- 1; ngridi <- 0} else {
        if(distr =="nb") { ngridi <- min(ngrid,yi)} else ngridi <- ngrid
        step <- yi/ngridi
        a<- rev(c(0,1:ngridi)+0.5)*step
      }
      
      #t=1
      t<-1
      convt <- pdist(a,ef[t],muT[t],disp[t]) - pdist(a-step,ef[t],muT[t],disp[t]) 
      if(nTnz >1){
        for(t in 2:nTnz){
          #t<-4
          at <- pdist(a,ef[t],muT[t],disp[t]) - pdist(a-step,ef[t],muT[t],disp[t]) ; 
          convo <- conv(convt,at) #sorted from high to low powers
          convti <- rev(rev(convo)[1:(ngridi+1)]) #sorted from high to low powers; only powers yi ... 0 are needed
        }} else {convti <- convt}
      allconvo <- c(allconvo,list(convti))
    }
    
    
    Ydens.loop <- function(i, mudispT){
      #i=1;j<-1;mudispT <- c(munaive,dispnaive);ngrid<-200
      #i=1;j<-1;mudispT <- c(2.7,0.9);ngrid<-25
      muTe <- mudispT[1]
      dispe <- mudispT[2] 
      yi <- as.numeric(datj[i])
      if(yi==0) {a <- 0.5; step <- 1; ngridi <- 0} else {
        if(distr =="nb") { ngridi <- min(ngrid,yi)} else ngridi <- ngrid
        step <- yi/ngridi
        a<- rev(c(0,1:ngridi)+0.5)*step
      }
      efsi <- allefs[1,i]+10^{-20}
      
      #t=1
      convt <- allconvo[[i]]
      at <- pdist(a, efsi, muTe,dispe) - pdist(a-step,efsi,muTe,dispe); 
      convo <- conv(convt,at) #sorted from high to low powers
      convt <- rev(rev(convo)[1:(ngridi+1)]) #sorted from high to low powers; only powers yi ... 0 are needed
      intpgf<-convt[1]/step
      #print(intpgf)
      return(max(10^(-100),intpgf)) 
    }
    
    loglik.loop <- function(mudispT){
      #mudisps <- param
      if(!prior) return(-sum(log(sapply(1:nsam, Ydens.loop, mudispT=mudispT)))) else {
        minusloglik <- -sum(log(sapply(1:nsam, Ydens.loop, mudispT=mudispT)))
        logpenal <- -logpenalty(j, mudispT=mudispT, priormu=priormu,priordisp=priordisp)
        return(minusloglik+logpenal) 
      }
    }
    
    if(is.null(startmat)){
    if(cleverinit){
      if(distr=="ln"){
      muinit <- mean(log(as.numeric(datj)))
      dispinit <- sd(log(as.numeric(datj)))
      initpar <- c(muinit,dispinit)
    } 
    if(distr=="nb"){
      mu0 <- mean(as.numeric(datj))
      muinit <- log(mu0)
      dispinit <- min(10,1/((var(as.numeric(datj))-mu0)/mu0^2))
      initpar <- c(muinit,dispinit)
      #initpar <- c(1,1)
    }
    } else {initpar <- c(1,1)}} 
    else {initpar <- startmat[j,]}
    cat("Initial:",round(loglik.loop(initpar),4),"Pars:",round(initpar,5))
    
    
    if(optimizer=="rsolnp") optres <- try(solnp(par = initpar, loglik.loop,  
                                                LB =  c(-1,lb), UB=c(20,ub),
                                                control=list(outer.iter=10) )) else {
                                                  optres <- try(optim(par = initpar, loglik.loop, lower =  c(-0.01,lb), upper=c(20,ub), 
                                                                      method = "L-BFGS-B"))  
                                                  # print(optres$value)
                                                  print(optres$par)
                                                }
    return(optres)
  } # end function j
  #resj(3)
  return(lapply(js,resj))
}

#' Estimation based on probability generating functions
estfPGF <- function(dat, is=1:ncol(dat), muTs, dispTs, ngrid=25, startmat=NULL,  sum2one=F, upper=1, distr= "ln"){
  
  #dat = dat; is = funknown; muTs = bothmuTs; dispTs = bothdispTs; ngrid=25; optimizer="rsolnp";noise=F; sum2one=T; upper=1; distr= "nb";munoise<- 0

  
  nT <- ncol(muTs)
  ngene <- nrow(muTs)
  
  if(distr =="ln") pdist <- function(x,f,mu,disp){return(plnorm(x,log(f) + mu, sd=disp))}
  if(distr =="nb") pdist <- function(x,f,mu,disp){return(pnbinom(x/f,mu=exp(mu),size=disp))} #because mu is searched on the log-scale
  
  
  resi <- function(i){
    #i<-55
    param <- rep(1/nT, nT)
    dati <- dat[,i]
    #j<-1
    print(i)
    Ydens.loop <- function(j, efs, muTs,dispTs){
      #j=2;efs <- rep(1/nT, nT) 
      muT <- muTs[j,]
      sdT <- dispTs[j,]
      
      yj <- dati[j]
      
      if(length(efs) == nT-1){
        #force last one to be 1-the rest
        efsum <- sum(efs)
        if(efsum>1) {return(10^(-100))} else {
          efsall <- c(efs,1-efsum)}} else { #length(efs) = T
            efsall <- efs
          }
      
      ef <- efsall
      if(yj==0) {a <- 0.5; step <- 1; ngridi <- 0} else {
        if(distr=="nb") { ngridi <- min(ngrid,yj)} else ngridi <- ngrid
        step <- yj/ngridi
        a<- rev(c(0,1:ngridi)+0.5)*step
      }
      
      #t=1
      t<-1
      convt <- pdist(a,ef[t],muT[t],sdT[t]) - pdist(a-step,ef[t],muT[t],sdT[t]) 
      
      for(t in 2:nT){
        #t<-2
        at <- pdist(a,ef[t],muT[t],sdT[t]) - pdist(a-step,ef[t],muT[t],sdT[t]); 
        convo <- conv(convt,at) #sorted from high to low powers
        convt <- rev(rev(convo)[1:(ngridi+1)]) #sorted from high to low powers; only powers yj ... 0 are needed
      }
      intpgf<-convt[1]/step
      intpgf
      return(max(10^(-100),intpgf)) 
    }
    
    # loglik.loop <- function(efs){
    #   return(-sum(log(sapply(1:ngene, Ydens.loop,efs=efs, muTs=muTs,dispTs=dispTs))))
    # }
    # 
    # loglik.loops <- function(efs){
    #   return(-log(sapply(1:ngene, Ydens.loop,efs=efs, muTs=muTs,dispTs=dispTs)))
    # }
    # 
    loglik.loop2 <- function(efslp){
      efs <- 1/(1+exp(-efslp))
      return(-sum(log(sapply(1:ngene, Ydens.loop,efs=efs, muTs=muTs,dispTs=dispTs))))
    }
    # loglik.loop(c(0.13782, 0.27888, 0.00001, 0.35854, 0.22474))
    #pmt <- proc.time()
    
    ef <- function(x) {sum(x[-1])-upper}; ub <- c(100,rep(1-0.00001,nT-1))
    if(is.null(startmat))  initpar <- param else initpar <- startmat[,i]
    # if(optimizer=="rsolnp") if(sum2one) optres <- try(solnp(par = initpar, loglik.loop, eqfun=ef, eqB =0,  LB = rep(0.00001,nT),UB=ub,
    #                                                         control=list(outer.iter=10))) else 
    #                                                           optres <- try(solnp(par = initpar, loglik.loop, ineqfun=ef, ineqUB = 0, ineqLB=-upper, LB = rep(0.00001,nT),UB=ub,
    #                                                                               control=list(outer.iter=10)))
    # else { optres <- try(optim(par = initpar[-1], loglik.loop, lower = rep(0.00001,nT-1),
    #                            upper=rep(1-0.00001,nT-1), method = "L-BFGS-B")); print(optres$par)
    optres <- try(optim(par = initpar[-1], loglik.loop2, method = "Brent", lower=-10,upper=10)); print(1/(1+exp(-optres$par)))
    return(optres)
  } # end function i
  #is <- 1:3;lapply(is,resi)
  return(lapply(is,resi))
}

#' Estimation based on probability generating functions
tunefpriorPGF <- function(tuningpar, dat, is=1:ncol(dat), meanfs, nsam, muTs, dispTs, ngrid=25, distr= "ln"){
  
  #dat = dat; is = funknown; muTs = bothmuTs; dispTs = bothdispTs; ngrid=25; optimizer="rsolnp";noise=F; 
  #sum2one=T; upper=1; distr= "nb";munoise<- 0
  
  #dat = dat; is = funknown; meanfs = mfsprior; nsam=100; muTs = bothmuTs; dispTs = bothdispTs;ngrid=25; distr= "nb"
  
  if(length(is) != nrow(meanfs)) {print("ERROR: length of argument 'is' should equal nr of rows of 'meanfs' "); 
    return(NULL)}
  
  nT <- ncol(muTs)
  ngene <- nrow(muTs)
  
  if(distr =="ln") pdist <- function(x,f,mu,disp){return(plnorm(x,log(f) + mu, sd=disp))}
  if(distr =="nb") pdist <- function(x,f,mu,disp){return(pnbinom(x/f,mu=exp(mu),size=disp))} #because mu is searched on the log-scale
  
  
  resi <- function(i){
    #i<-1
    ind <- is[i]
    param <- rep(1/nT, nT)
    dati <- dat[,ind]
    #j<-1
    print(i)
    Ydens.loop <- function(j, efs, muTs,dispTs){
      #j=2;efs <- rep(1/nT, nT) 
      muT <- muTs[j,]
      sdT <- dispTs[j,]
      
      yj <- dati[j]
      
      if(length(efs) == nT-1){
        #force last one to be 1-the rest
        efsum <- sum(efs)
        if(efsum>1) {return(10^(-100))} else {
          efsall <- c(efs,1-efsum)}} else { #length(efs) = T
            efsall <- efs
          }
      
      ef <- efsall
      if(yj==0) {a <- 0.5; step <- 1; ngridi <- 0} else {
        if(distr=="nb") { ngridi <- min(ngrid,yj)} else ngridi <- ngrid
        step <- yj/ngridi
        a<- rev(c(0,1:ngridi)+0.5)*step
      }
      
      #t=1
      t<-1
      convt <- pdist(a,ef[t],muT[t],sdT[t]) - pdist(a-step,ef[t],muT[t],sdT[t]) 
      
      for(t in 2:nT){
        #t<-2
        at <- pdist(a,ef[t],muT[t],sdT[t]) - pdist(a-step,ef[t],muT[t],sdT[t]); 
        convo <- conv(convt,at) #sorted from high to low powers
        convt <- rev(rev(convo)[1:(ngridi+1)]) #sorted from high to low powers; only powers yj ... 0 are needed
      }
      intpgf<-convt[1]/step
      intpgf
      return(max(10^(-100),intpgf)) 
    }
    
    # loglik.loop <- function(efs){
    #   return(-sum(log(sapply(1:ngene, Ydens.loop,efs=efs, muTs=muTs,dispTs=dispTs))))
    # }
    # 
    # loglik.loops <- function(efs){
    #   return(-log(sapply(1:ngene, Ydens.loop,efs=efs, muTs=muTs,dispTs=dispTs)))
    # }
    # 
    loglik.loop2 <- function(efs){
      return(sum(log(sapply(1:ngene, Ydens.loop,efs=efs, muTs=muTs,dispTs=dispTs))))
    }
    
    #nsam <- 100;tuningpar=10;
    efssam0 <- rdirichlet(nsam,alpha=tuningpar*meanfs[i,])
    minimax <- function(x) if(x<=10^(-8)) return(10^(-8)) else return(x)
    efssam <- apply(efssam0,c(1,2),minimax)
    
    logliks <- apply(efssam, 1,loglik.loop2)
    
    maxll <- max(logliks)
    
    logmlest <- maxll + log(sum(exp(logliks - maxll)))
    return(logmlest)
  }
  meanll <- -mean(sapply(1:length(is),resi))
  print(meanll)
  return(meanll)
}


postsum <- function(t,i,j,dat, fs, muTs, dispTs, M=1000,ngrid=25,distr= "ln"){
  #i <- 2;j<-2;dat <- dat; t <- 2; M<-1000; ngrid<-25;distr <- "nb"; muTs <- bothmuTs; dispTs <- bothdispTs
  yij <- as.numeric(dat[j,i])  #gene in row, sample in column
  #yij
  if(yij==0){
    postres <- c(postmn = 0,postsd=10^{-8},postmnlog = -100,postsdlog = 10)
    return(postres)
  } else {
  
  fsi <- as.numeric(fs[,i])
  fsit <- fsi[t]
  if(fsit>=0.99) return(c(postmn = yij,postdisp=0,postmnlog = log(yij),postdisplog = 0))
  if(fsit<=0.01) return(c(postmn = NA,postdisp=NA,postmnlog = NA,postdisplog = NA))
  
  if(distr =="ln") pdist <- function(x,f,mu,disp) return(plnorm(x,log(f) + mu, sd=disp))
  if(distr =="nb") pdist <- function(x,f,mu,disp) return(pnbinom(x/f,mu=exp(mu),size=disp))
  
  #fsi
  if(fsit > 0.01 & fsit < 0.99){
    musj <- as.numeric(muTs[j,])
    #musj
    dispsj <- as.numeric(dispTs[j,])
    #dispsj
    
    muijt <- log(fsi[t]) + musj[t]
    dispijt <- dispsj[t]
    
    
    if(distr =="ln") sam <- rtrunc(M,spec="lnorm", b=yij,mean=muijt,disp=dispijt)
    if(distr =="nb") sam <- rtrunc(M,spec="nbinom",b=yij, mu=exp(muijt), size = dispijt)
    
    efsi <- fsi[-t] #t+1 because noise is the fist term
    musjb <-musj[-t]
    dispsjb <- dispsj[-t]
    
    
    whnz <- which(efsi != 0)
    lf <- efsi[whnz]
    muT <- musjb[whnz]
    dispT <- dispsjb[whnz]
    nTnz <- length(whnz)
    
    step <- yij/ngrid
    a<- rev(c(0,1:ngrid)+0.5)*step
    
    #t=1
    t<-1
    convt <- pdist(a,lf[t],muT[t],dispT[t]) - pdist(a-step,lf[t],muT[t],dispT[t]) 
    if(nTnz >1){
      for(t in 2:nTnz){
        #t<-2
        at <- pdist(a,lf[t],muT[t],dispT[t]) - pdist(a-step,lf[t],muT[t],dispT[t])  
        convo <- conv(convt,at) #sorted from high to low powers
        convt <- rev(rev(convo)[1:(ngrid+1)]) #sorted from high to low powers; only powers yi ... 0 are needed
      }
    }
    convtrev <- rev(convt)
    # probk <- function(k, convtrev, step){
    #   wh <- ceiling(k/step)
    #   wei <- convtrev[wh]
    #   return(wei)
    # }
    
    
    wh <- ceiling((yij-sam + 10^(-10))/step) 
    wei0 <- convtrev[wh]
    wei <- wei0/sum(wei0)
    if(sum(wei0) <= 1) postres = c(postmn = NA,postsd=NA,postmnlog = NA,postsdlog = NA) else {
      samdivf <- sam/fsi[t] 
      samdivf2 <- samdivf^2 
      if(distr=="nb") lsamdivf <- log(samdivf+0.5) else lsamdivf <- log(samdivf)
      lsamdivf2 <- lsamdivf^2 
      postmeans <- c(samdivf %*% wei,samdivf2 %*% wei,lsamdivf %*% wei, lsamdivf2 %*% wei) 
      postsds <- c(sqrt(postmeans[2]-postmeans[1]^2),sqrt(postmeans[4]-postmeans[3]^2))
      postres <- c(postmn = postmeans[1],postsd=postsds[1],postmnlog = postmeans[3],postsdlog = postsds[2])
    }
    return(postres)
  }
  }
}


