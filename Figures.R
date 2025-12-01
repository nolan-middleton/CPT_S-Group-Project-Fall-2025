########################################################################
# This R script compiles all the data for the plots.                   #
########################################################################

##### SETUP #####
require(ggplot2)
require(ggforce)
require(jsonlite)
require(stringr)
require(rstudioapi)
require(grid)

# Directories
setwd(dirname(getActiveDocumentContext()$path))

if (!dir.exists("Figures")) {
  dir.create("Figures")
}

# Plotting
plotTheme <- theme(
  plot.margin = margin(t = 10, r = 10, b = 10, l = 10, unit = "pt"),
  plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
  panel.background = element_blank(),
  panel.grid = element_blank(),
  axis.line = element_line(linewidth = 1.5, lineend = "square"),
  axis.ticks = element_line(linewidth = 1.5, color = "black"),
  axis.ticks.length = unit(7.5, "pt"),
  axis.title = element_text(face = "bold", size = 14),
  axis.text = element_text(face = "bold", size = 12, color = "black"),
  legend.text = element_text(face = "bold", size = 12, color = "black"),
  legend.justification = c("left", "top"),
  legend.direction = "vertical",
  legend.margin = margin(),
  legend.title = element_text(face = "bold", size = 12, color = "black")
)

# Load data
datasets <- unlist(read.delim("datasets.txt", header = FALSE))
names(datasets) <- c("UCC", "SCLC", "SEC", "MDS", "ALL", "HIV", "JIA",
                     "GBM", "MDG")
models <- c("DecisionTree", "kNearestNeighbours", "NaiveBayes",
            "RandomForest", "SupportVectorMachine")

load_model <- function(L) {
  if ("results" %in% names(L)) {
    return(matrix(unlist(L$results),nrow=length(L$results),byrow=TRUE))
  } else {
    R <- list()
    for (i in 1:length(L)) {
      R[[names(L)[i]]] <- load_model(L[[i]])
    }
    return(R)
  }
}

data <- list()
for (dataset in datasets) {
  print(paste0("> ", dataset, "..."))
  data[[dataset]] <- list()
  for (i in 1:5) {
    print(paste0(">> Strategy ", toString(i), "..."))
    data[[dataset]][[i]] <- list()
    if (i == 1) {
      for (model in models) {
        print(paste0(">>> ", model, "..."))
        data[[dataset]][[i]][[model]] <- load_model(
          read_json(paste0(dataset, "/", toString(i), "/", model, ".json"))
        )
      }
    } else {
      subdirs <- list.dirs(paste0(dataset,"/",toString(i)),full.names=FALSE)
      for (dir in subdirs[2:length(subdirs)]) {
        print(paste0(">>> ", dir, "..."))
        data[[dataset]][[i]][[dir]] <- list()
        for (model in models) {
          data[[dataset]][[i]][[dir]][[model]] <- load_model(
            read_json(
              paste0(dataset,"/",toString(i),"/",dir,"/",model,".json")
            )
          )
        }
      }
    }
  }
}

# Save this object so we don't have to run that loop again
save(data, file = "Figures/dataObj.rda")

# Load the data
data <- load("Figures/dataObj.rda")

# Other functions
acc <- function(mat) {
  return(sum(diag(mat)) / sum(mat))
}

GO_cats <- c("Component", "Function", "Process")

make_barplots <- function(fig_function, data, model, w=2000, h=1000) {
  if (!dir.exists(paste0("Figures/", model))) {
    dir.create(paste0("Figures/", model))
  }
  
  # Baseline Performance
  baselineData <- list()
  for (dataset in datasets) {
    baselineData[[dataset]] <- data[[dataset]][[1]][[model]]
  }
  
  P <- fig_function(baselineData, "No Dimensionality Reduction")
  
  ggsave(
    paste0("Figures/", model, "/1.png"),
    plot = P,
    width = w,
    height = h,
    units = "px"
  )
  
  # Strategy 2 Performance
  for (cat in GO_cats) {
    thisData <- list()
    for (dataset in datasets) {
      thisData[[dataset]] <- data[[dataset]][[2]][[cat]][[model]]
    }
    
    P <- fig_function(thisData, paste0("Aggregation by ", cat))
    
    ggsave(
      paste0("Figures/", model, "/2_", cat, ".png"),
      plot = P,
      width = w,
      height = h,
      units = "px"
    )
  }
  
  # Strategy 3 Performance
  for (d in 2:9) {
    thisData <- list()
    for (dataset in datasets) {
      thisData[[dataset]] <- data[[dataset]][[3]][[toString(d)]][[model]]
    }
    
    P <- fig_function(thisData, paste0("PCA d=", toString(d)))
    
    ggsave(
      paste0("Figures/", model, "/3_", toString(d), ".png"),
      plot = P,
      width = w,
      height = h,
      units = "px"
    )
  }
  
  # Strategy 4 Performance
  for (d in 2:9) {
    thisData <- list()
    for (dataset in datasets) {
      thisData[[dataset]] <- data[[dataset]][[4]][[toString(d)]][[model]]
    }
    
    P <- fig_function(thisData, paste0("Kernelized PCA d=", toString(d)))
    
    ggsave(
      paste0("Figures/", model, "/4_", toString(d), ".png"),
      plot = P,
      width = w,
      height = h,
      units = "px"
    )
  }
  
  # Strategy 5 Performance
  for (d in 2:9) {
    thisData <- list()
    for (dataset in datasets) {
      thisData[[dataset]] <- data[[dataset]][[5]][[toString(d)]][[model]]
    }
    
    P <- fig_function(thisData, paste0("NMF d=", toString(d)))
    
    ggsave(
      paste0("Figures/", model, "/5_", toString(d), ".png"),
      plot = P,
      width = w,
      height = h,
      units = "px"
    )
  }
}

##### DECISION TREE FIGURES #####

DT_fig <- function(dataList, title = NULL) {
  plotData <- data.frame(
    dataset = numeric(0),
    depth = numeric(0),
    accuracy = numeric(0)
  )
  for (i in 1:length(dataList)) {
    for (depth in names(dataList[[i]])) {
      plotData <- rbind(
        plotData,
        data.frame(
          dataset = i,
          depth = strtoi(depth),
          accuracy = acc(dataList[[i]][[depth]])
        )
      )
    }
  }
  
  P <- ggplot(data = plotData) +
    geom_col(
      mapping = aes(
        x = dataset,
        y = accuracy,
        fill = as.factor(depth)
      ),
      position = position_dodge2(padding = 0),
      color = "black"
    ) +
    plotTheme +
    scale_fill_discrete(
      name = "Depth",
      type = c("white", "grey", "black")
    ) +
    scale_x_continuous(
      name = "Dataset",
      breaks = 1:length(datasets),
      limits = c(0.5,length(datasets) + 0.5),
      labels = names(datasets)
    ) +
    theme(axis.text.x = element_text(angle = 30, hjust = 1)) +
    scale_y_continuous(
      name = "Accuracy",
      breaks = 0:5 / 5,
      limits = c(0,1),
      expand = c(0,0)
    )
  
  if (!is.null(title)) {
    P <- P + ggtitle(title)
  }
  
  return(P)
}

make_barplots(DT_fig, data, "DecisionTree")

##### K-NEAREST NEIGHBOURS FIGURES #####

kNN_fig <- function(dataList, title = NULL) {
  plotData <- data.frame(
    dataset = numeric(0),
    k = numeric(0),
    p = numeric(0),
    accuracy = numeric(0)
  )
  for (i in 1:length(dataList)) {
    for (k in names(dataList[[i]])) {
      for (p in names(dataList[[i]][[k]])) {
        plotData <- rbind(
          plotData,
          data.frame(
            dataset = i,
            k = strtoi(k),
            p = strtoi(p),
            accuracy = acc(dataList[[i]][[k]][[p]])
          )
        )
      }
    }
  }
  
  label_list <- names(datasets)
  names(label_list) <- 1:length(datasets)
  
  P <- ggplot(data = plotData) +
    geom_col(
      mapping = aes(
        x = p,
        y = accuracy,
        fill = as.factor(k)
      ),
      position = position_dodge2(padding = 0),
      color = "black",
      linewidth = 0.25
    ) +
    facet_grid(
      . ~ dataset,
      space = "free_x",
      scales = "free_x",
      labeller = as_labeller(label_list)
    ) +
    plotTheme +
    scale_fill_discrete(
      name = "k",
      type = c("white", "grey", "black")
    ) +
    scale_x_continuous(
      name = "p",
      breaks = 1:max(plotData$p),
      limits = c(0.5, max(plotData$p) + 0.5)
    ) +
    scale_y_continuous(
      name = "Accuracy",
      breaks = 0:5 / 5,
      limits = c(0,1),
      expand = c(0,0)
    ) +
    theme(strip.text = element_text(face = "bold"))
  
  if (!is.null(title)) {
    P <- P + ggtitle(title)
  }
  
  return(P)
}

make_barplots(kNN_fig, data, "kNearestNeighbours")

##### NAIVE BAYES FIGURES #####

NB_fig <- function(dataList, title = NULL) {
  plotData <- data.frame(
    dataset = numeric(0),
    accuracy = numeric(0)
  )
  for (i in 1:length(dataList)) {
    plotData <- rbind(
      plotData,
      data.frame(
        dataset = i,
        accuracy = acc(dataList[[i]])
      )
    )
  }
  
  P <- ggplot(data = plotData) +
    geom_col(
      mapping = aes(
        x = dataset,
        y = accuracy,
      ),
      fill = "grey",
      color = "black"
    ) +
    plotTheme +
    scale_x_continuous(
      name = "Dataset",
      breaks = 1:length(datasets),
      limits = c(0.5,length(datasets) + 0.5),
      labels = names(datasets)
    ) +
    theme(axis.text.x = element_text(angle = 30, hjust = 1)) +
    scale_y_continuous(
      name = "Accuracy",
      breaks = 0:5 / 5,
      limits = c(0,1),
      expand = c(0,0)
    )
  
  if (!is.null(title)) {
    P <- P + ggtitle(title)
  }
  
  return(P)
}

make_barplots(NB_fig, data, "NaiveBayes")

##### RANDOM FOREST FIGURES #####

RF_fig <- function(dataList, title = NULL) {
  plotData <- data.frame(
    dataset = numeric(0),
    depth = numeric(0),
    n_estimators = numeric(0),
    accuracy = numeric(0)
  )
  for (i in 1:length(dataList)) {
    for (depth in names(dataList[[i]])) {
      for (n_estimators in names(dataList[[i]][[depth]])) {
        N <- 1
        if (n_estimators == "200") { N <- 2 }
        else if (n_estimators == "500") { N <- 3 }
        plotData <- rbind(
          plotData,
          data.frame(
            dataset = i,
            depth = strtoi(depth),
            n_estimators = N,
            accuracy = acc(
              dataList[[i]][[depth]][[n_estimators]]
            )
          )
        )
      }
    }
  }
  
  label_list <- names(datasets)
  names(label_list) <- 1:length(datasets)
  
  P <- ggplot(data = plotData) +
    geom_col(
      mapping = aes(
        x = n_estimators,
        y = accuracy,
        fill = as.factor(depth)
      ),
      position = position_dodge2(padding = 0),
      color = "black",
      linewidth = 0.25
    ) +
    facet_grid(
      . ~ dataset,
      space = "free_x",
      scales = "free_x",
      labeller = as_labeller(label_list)
    ) +
    plotTheme +
    scale_fill_discrete(
      name = "Depth",
      type = c("white", "grey", "black")
    ) +
    scale_x_continuous(
      name = "Number of Estimators",
      breaks = 1:3,
      limits = c(0.5, 3.5),
      labels = c("100", "200", "500")
    ) +
    scale_y_continuous(
      name = "Accuracy",
      breaks = 0:5 / 5,
      limits = c(0,1),
      expand = c(0,0)
    ) +
    theme(
      strip.text = element_text(face = "bold"),
      axis.text.x = element_text(angle=45, hjust=1.25, vjust=1.5, size=10)
    )
  
  if (!is.null(title)) {
    P <- P + ggtitle(title)
  }
  
  return(P)
}

make_barplots(RF_fig, data, "RandomForest")

##### SUPPORT VECTOR MACHINE FIGURES #####

SVM_fig <- function(dataList, title = NULL) {
  plotData <- data.frame(
    dataset = numeric(0),
    kernel = numeric(0),
    C = numeric(0),
    accuracy = numeric(0)
  )
  for (i in 1:length(dataList)) {
    for (kernel in names(dataList[[i]])) {
      for (C in names(dataList[[i]][[kernel]])) {
        K <- 1
        if (kernel == "linear") { K <- 2 }
        if (kernel == "quadratic") { K <- 3 }
        if (kernel == "cubic") { K <- 4 }
        plotData <- rbind(
          plotData,
          data.frame(
            dataset = i,
            kernel = K,
            C = as.numeric(C),
            accuracy = acc(dataList[[i]][[kernel]][[C]])
          )
        )
      }
    }
  }
  
  label_list <- names(datasets)
  names(label_list) <- 1:length(datasets)
  
  P <- ggplot(data = plotData) +
    geom_col(
      mapping = aes(
        x = kernel,
        y = accuracy,
        fill = as.factor(C)
      ),
      position = position_dodge2(padding = 0),
      color = "black",
      linewidth = 0.25
    ) +
    facet_grid(
      . ~ dataset,
      space = "free_x",
      scales = "free_x",
      labeller = as_labeller(label_list)
    ) +
    plotTheme +
    scale_fill_discrete(
      name = "log10(C)",
      type = c("white", "#D2D2D2", "#A8A8A8", "#7E7E7E", "#545454",
               "#2A2A2A", "black"),
      labels = -3:3
    ) +
    scale_x_continuous(
      breaks = 1:4,
      limits = c(0.5, 4.5),
      labels = c("rbf", "linear", "quadratic", "cubic"),
      name = "Kernel"
    ) +
    scale_y_continuous(
      name = "Accuracy",
      breaks = 0:5 / 5,
      limits = c(0,1),
      expand = c(0,0)
    ) +
    theme(
      strip.text = element_text(face = "bold"),
      axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)
    )
  
  if (!is.null(title)) {
    P <- P + ggtitle(title)
  }
  
  return(P)
}

make_barplots(SVM_fig, data, "SupportVectorMachine", 2500, 1250)

##### LABELS #####

labels <- list()
for (dataset in datasets) {
  L <- unlist(read.delim(paste0(dataset, "/metadata.txt")))
  names(L) <- c()
  L <- unlist(strsplit(L, ": "))[2]
  L <- unlist(strsplit(L, ", "))
  L <- unlist(strsplit(L, ".tsv"))
  
  labels[[dataset]] <- L
}

nice_labels <- list(
  "UlcerativeColitisAndCrohns" = list(
    "title" = "Disease",
    "labels" = c("Normal", "Ulcerative\nColitis", "Crohn's\nDisease")
  ),
  "SquamousCellLungCarcinomas" = list(
    "title" = "Stage",
    "labels" = c("IA", "IB", "IIA", "IIB", "IIIA", "IIIB")
  ),
  "SmokerEpithelialCells" = list(
    "title" = "Cancer",
    "labels" = c("None", "Confirmed", "Suspected")
  ),
  "MDS" = list(
    "title" = "Disease",
    "labels" = c("Myelodysplastic\nSyndrome", "Healthy")
  ),
  "PediatricALL" = list(
    "title" = "Relapse",
    "labels" = c("Early", "Late", "None")
  ),
  "HIV" = list(
    "title" = "HIV",
    "labels" = c("Negative", "Positive")
  ),
  "JuvenileIdiopathicArthritis" = list(
    "title" = "JIA",
    "labels" = c("Systemic", "Non-Systemic", "None")
  ),
  "Glioblastoma" = list(
    "title" = "Overall\nSurvival",
    "labels" = c("Short-Term", "Intermediate", "Long-Term")
  ),
  "MacularDegeneration" = list(
    "title" = "Disease",
    "labels" = c("Normal", "Macular\nDegeneration")
  )
)

##### SCATTER PLOTS #####

make_scatter <- function(X, dirname, xlab, ylab) {
  if (!dir.exists(dirname)) {
    dir.create(dirname)
  }
  
  # UlcerativeColitisAndCrohns
  P <- ggplot(data = X[[1]]) +
    geom_point(
      mapping = aes(
        x = x,
        y = y,
        fill = factor(label,levels = nice_labels[[1]]$labels)
      ),
      shape = 21
    ) +
    plotTheme +
    scale_x_continuous(name = xlab) +
    scale_y_continuous(name = ylab) +
    scale_fill_discrete(
      name = nice_labels[[1]]$title,
      type = c("white", "grey", "black")
    )
  
  ggsave(
    paste0(dirname, "/", datasets[1], ".png"),
    plot = P,
    width = 2000,
    height = 1000,
    units = "px"
  )
  
  # SquamousCellLungCarcinomas
  P <- ggplot(data = X[[2]]) +
    geom_point(
      mapping = aes(
        x = x,
        y = y,
        fill = factor(label, levels = nice_labels[[2]]$labels),
        shape = factor(label, levels = nice_labels[[2]]$labels)
      )
    ) +
    plotTheme +
    scale_x_continuous(name = xlab) +
    scale_y_continuous(name = ylab) +
    scale_fill_manual(
      name = nice_labels[[2]]$title,
      labels = nice_labels[[2]]$labels,
      values = rep(c("white", "grey", "black"), each = 2)
    ) +
    scale_shape_manual(
      name = nice_labels[[2]]$title,
      labels = nice_labels[[2]]$labels,
      values = rep(c(21, 22), 3)
    )
  
  ggsave(
    paste0(dirname, "/", datasets[2], ".png"),
    plot = P,
    width = 2000,
    height = 1000,
    units = "px"
  )
  
  # SmokerEpithelialCells
  P <- ggplot(data = X[[3]]) +
    geom_point(
      mapping = aes(
        x = x,
        y = y,
        fill = factor(label,levels = nice_labels[[3]]$labels)
      ),
      shape = 21
    ) +
    plotTheme +
    scale_x_continuous(name = xlab) +
    scale_y_continuous(name = ylab) +
    scale_fill_discrete(
      name = nice_labels[[3]]$title,
      type = c("white", "grey", "black")
    )
  
  ggsave(
    paste0(dirname, "/", datasets[3], ".png"),
    plot = P,
    width = 2000,
    height = 1000,
    units = "px"
  )
  
  # MDS
  P <- ggplot(data = X[[4]]) +
    geom_point(
      mapping = aes(
        x = x,
        y = y,
        fill = factor(label,levels = nice_labels[[4]]$labels)
      ),
      shape = 21
    ) +
    plotTheme +
    scale_x_continuous(name = xlab) +
    scale_y_continuous(name = ylab) +
    scale_fill_discrete(
      name = nice_labels[[4]]$title,
      type = c("white", "black")
    )
  
  ggsave(
    paste0(dirname, "/", datasets[4], ".png"),
    plot = P,
    width = 2000,
    height = 1000,
    units = "px"
  )
  
  # PediatricALL
  P <- ggplot(data = X[[5]]) +
    geom_point(
      mapping = aes(
        x = x,
        y = y,
        fill = factor(label,levels = nice_labels[[5]]$labels)
      ),
      shape = 21
    ) +
    plotTheme +
    scale_x_continuous(name = xlab) +
    scale_y_continuous(name = ylab) +
    scale_fill_discrete(
      name = nice_labels[[5]]$title,
      type = c("white", "grey", "black")
    )
  
  ggsave(
    paste0(dirname, "/", datasets[5], ".png"),
    plot = P,
    width = 2000,
    height = 1000,
    units = "px"
  )
  
  # MDS
  P <- ggplot(data = X[[6]]) +
    geom_point(
      mapping = aes(
        x = x,
        y = y,
        fill = factor(label,levels = nice_labels[[6]]$labels)
      ),
      shape = 21
    ) +
    plotTheme +
    scale_x_continuous(name = xlab) +
    scale_y_continuous(name = ylab) +
    scale_fill_discrete(
      name = nice_labels[[6]]$title,
      type = c("white", "black")
    )
  
  ggsave(
    paste0(dirname, "/", datasets[6], ".png"),
    plot = P,
    width = 2000,
    height = 1000,
    units = "px"
  )
  
  # JuvenileIdiopathicArthritis
  P <- ggplot(data = X[[7]]) +
    geom_point(
      mapping = aes(
        x = x,
        y = y,
        fill = factor(label,levels = nice_labels[[7]]$labels)
      ),
      shape = 21
    ) +
    plotTheme +
    scale_x_continuous(name = xlab) +
    scale_y_continuous(name = ylab) +
    scale_fill_discrete(
      name = nice_labels[[7]]$title,
      type = c("white", "grey", "black")
    )
  
  ggsave(
    paste0(dirname, "/", datasets[7], ".png"),
    plot = P,
    width = 2000,
    height = 1000,
    units = "px"
  )
  
  # Glioblastoma
  P <- ggplot(data = X[[8]]) +
    geom_point(
      mapping = aes(
        x = x,
        y = y,
        fill = factor(label,levels = nice_labels[[8]]$labels)
      ),
      shape = 21
    ) +
    plotTheme +
    scale_x_continuous(name = xlab) +
    scale_y_continuous(name = ylab) +
    scale_fill_discrete(
      name = nice_labels[[8]]$title,
      type = c("white", "grey", "black")
    )
  
  ggsave(
    paste0(dirname, "/", datasets[8], ".png"),
    plot = P,
    width = 2000,
    height = 1000,
    units = "px"
  )
  
  # MacularDegeneration
  P <- ggplot(data = X[[9]]) +
    geom_point(
      mapping = aes(
        x = x,
        y = y,
        fill = factor(label,levels = nice_labels[[9]]$labels)
      ),
      shape = 21
    ) +
    plotTheme +
    scale_x_continuous(name = xlab) +
    scale_y_continuous(name = ylab) +
    scale_fill_discrete(
      name = nice_labels[[9]]$title,
      type = c("white", "black")
    )
  
  ggsave(
    paste0(dirname, "/", datasets[9], ".png"),
    plot = P,
    width = 2000,
    height = 1000,
    units = "px"
  )
}

# PCA
X <- list()
for (dataset in datasets) {
  X[[dataset]] <- read.delim(
    paste0(dataset, "/3/2/training_X.tsv"),
    header = FALSE
  )
  X[[dataset]] <- cbind(
    X[[dataset]],
    read.delim(paste0(dataset, "/3/2/training_Y.tsv"), header = FALSE)
  )
  colnames(X[[dataset]]) <- c("x", "y", "label")
  X[[dataset]]$label <- nice_labels[[dataset]]$labels[X[[dataset]]$label+1]
}

make_scatter(
  X,
  "Figures/PCA",
  "Principle Component 1",
  "Principle Component 2"
)

# Kernelized PCA
X <- list()
for (dataset in datasets) {
  X[[dataset]] <- read.delim(
    paste0(dataset, "/4/2/training_X.tsv"),
    header = FALSE
  )
  X[[dataset]] <- cbind(
    X[[dataset]],
    read.delim(paste0(dataset, "/4/2/training_Y.tsv"), header = FALSE)
  )
  colnames(X[[dataset]]) <- c("x", "y", "label")
  X[[dataset]]$label <- nice_labels[[dataset]]$labels[X[[dataset]]$label+1]
}

make_scatter(
  X,
  "Figures/kernelizedPCA",
  "Principle Component 1",
  "Principle Component 2"
)

# NMF
X <- list()
for (dataset in datasets) {
  X[[dataset]] <- read.delim(
    paste0(dataset, "/5/2/training_X.tsv"),
    header = FALSE
  )
  X[[dataset]] <- cbind(
    X[[dataset]],
    read.delim(paste0(dataset, "/5/2/training_Y.tsv"), header = FALSE)
  )
  colnames(X[[dataset]]) <- c("x", "y", "label")
  X[[dataset]]$label <- nice_labels[[dataset]]$labels[X[[dataset]]$label+1]
}

make_scatter(
  X,
  "Figures/NMF",
  "Feature 1",
  "Feature 2"
)
