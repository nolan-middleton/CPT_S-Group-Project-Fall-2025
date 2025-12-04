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
require(ggh4x)
require(ggpattern)

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
  legend.title = element_text(face = "bold", size = 12, color = "black"),
  strip.text = element_text(face = "bold", size = 12, color = "black"),
  strip.background = element_blank()
)

# Variables
datasets <- unlist(read.delim("datasets.txt", header = FALSE))
names(datasets) <- c("UCC", "SCLC", "SEC", "MDS", "ALL", "HIV", "JIA",
                     "GBM", "MDG")
models <- c("DecisionTree", "kNearestNeighbours", "NaiveBayes",
            "RandomForest", "SupportVectorMachine")

strategies <- c("No", "Aggregate", "PCA", "Kernelized PCA", "NMF")
strategy_titles <- list(
  function(dim) { return("No Dimensionality Reduction") },
  function(dim) { return(paste0("Aggregation by ", dim)) },
  function(dim) { return(paste0("PCA (Components = ", dim, ")")) },
  function(dim) { return(paste0("Kernelized PCA (Components = ", dim, ")")) },
  function(dim) { return(paste0("NMF (Features = ", dim, ")")) }
)

GO_cats <- c("Component", "Function", "Process")

PCA_dims <- 2:9

# Functions
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

acc <- function(mat) {
  return(sum(diag(mat)) / sum(mat))
}

# Load data
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

D <- list(
  DecisionTree = data.frame(
    dataset = character(0),
    strategy = character(0),
    dim = character(0),
    depth = numeric(0),
    accuracy = numeric(0)
  ),
  kNearestNeighbours = data.frame(
    dataset = character(0),
    strategy = character(0),
    dim = character(0),
    k = numeric(0),
    p = numeric(0),
    accuracy = numeric(0)
  ),
  NaiveBayes = data.frame(
    dataset = character(0),
    strategy = character(0),
    dim = character(0),
    depth = character(0),
    accuracy = numeric(0)
  ),
  RandomForest = data.frame(
    dataset = character(0),
    strategy = character(0),
    dim = character(0),
    depth = numeric(0),
    n_estimators = numeric(0),
    accuracy = numeric(0)
  ),
  SupportVectorMachine = data.frame(
    dataset = character(0),
    strategy = character(0),
    dim = character(0),
    kernel = character(0),
    C = numeric(0),
    accuracy = numeric(0)
  )
)
for (i in 1:length(datasets)) {
  dataset = datasets[i]
  for (depth in names(data[[dataset]][[1]]$DecisionTree)) {
    D$DecisionTree <- rbind(
      D$DecisionTree,
      data.frame(
        dataset = names(dataset),
        strategy = strategies[1],
        dim = "",
        depth = strtoi(depth),
        accuracy = acc(data[[dataset]][[1]]$DecisionTree[[depth]])
      )
    )
  }
  for (k in names(data[[dataset]][[1]]$kNearestNeighbours)) {
    for (p in names(data[[dataset]][[1]]$kNearestNeighbours[[k]])) {
      D$kNearestNeighbours <- rbind(
        D$kNearestNeighbours,
        data.frame(
          dataset = names(dataset),
          strategy = strategies[1],
          dim = "",
          k = strtoi(k),
          p = strtoi(p),
          accuracy = acc(data[[dataset]][[1]]$kNearestNeighbours[[k]][[p]])
        )
      )
    }
  }
  D$NaiveBayes <- rbind(
    D$NaiveBayes,
    data.frame(
      dataset = names(dataset),
      strategy = strategies[1],
      dim = "",
      accuracy = acc(data[[dataset]][[1]]$NaiveBayes)
    )
  )
  for (depth in names(data[[dataset]][[1]]$RandomForest)) {
    for (n in names(data[[dataset]][[1]]$RandomForest[[depth]])) {
      D$RandomForest <- rbind(
        D$RandomForest,
        data.frame(
          dataset = names(dataset),
          strategy = strategies[1],
          dim = "",
          depth = strtoi(depth),
          n_estimators = strtoi(n),
          accuracy = acc(data[[dataset]][[1]]$RandomForest[[depth]][[n]])
        )
      )
    }
  }
  for (kernel in names(data[[dataset]][[1]]$SupportVectorMachine)) {
    for (C in names(data[[dataset]][[1]]$SupportVectorMachine[[kernel]])) {
      D$SupportVectorMachine <- rbind(
        D$SupportVectorMachine,
        data.frame(
          dataset = names(dataset),
          strategy = strategies[1],
          dim = "",
          kernel = kernel,
          C = as.numeric(C),
          accuracy = acc(
            data[[dataset]][[1]]$SupportVectorMachine[[kernel]][[C]]
          )
        )
      )
    }
  }
  
  for (S in 2:5) {
    for (dim in names(data[[dataset]][[S]])) {
      for (depth in names(data[[dataset]][[S]][[dim]]$DecisionTree)) {
        D$DecisionTree <- rbind(
          D$DecisionTree,
          data.frame(
            dataset = names(dataset),
            strategy = strategies[S],
            dim = dim,
            depth = strtoi(depth),
            accuracy = acc(data[[dataset]][[S]][[dim]]$DecisionTree[[depth]])
          )
        )
      }
      for (k in names(data[[dataset]][[S]][[dim]]$kNearestNeighbours)) {
        for (p in names(data[[dataset]][[S]][[dim]]$kNearestNeighbours[[k]])) {
          D$kNearestNeighbours <- rbind(
            D$kNearestNeighbours,
            data.frame(
              dataset = names(dataset),
              strategy = strategies[S],
              dim = dim,
              k = strtoi(k),
              p = strtoi(p),
              accuracy = acc(
                data[[dataset]][[S]][[dim]]$kNearestNeighbours[[k]][[p]]
              )
            )
          )
        }
      }
      D$NaiveBayes <- rbind(
        D$NaiveBayes,
        data.frame(
          dataset = names(dataset),
          strategy = strategies[S],
          dim = dim,
          accuracy = acc(data[[dataset]][[S]][[dim]]$NaiveBayes)
        )
      )
      for (depth in names(data[[dataset]][[S]][[dim]]$RandomForest)) {
        for (n in names(data[[dataset]][[S]][[dim]]$RandomForest[[depth]])) {
          D$RandomForest <- rbind(
            D$RandomForest,
            data.frame(
              dataset = names(dataset),
              strategy = strategies[S],
              dim = dim,
              depth = strtoi(depth),
              n_estimators = strtoi(n),
              accuracy = acc(
                data[[dataset]][[S]][[dim]]$RandomForest[[depth]][[n]]
              )
            )
          )
        }
      }
      for (kernel in names(data[[dataset]][[S]][[dim]]$SupportVectorMachine)) {
        for (C in names(
          data[[dataset]][[S]][[dim]]$SupportVectorMachine[[kernel]])
        ) {
          D$SupportVectorMachine <- rbind(
            D$SupportVectorMachine,
            data.frame(
              dataset = names(dataset),
              strategy = strategies[S],
              dim = dim,
              kernel = kernel,
              C = as.numeric(C),
              accuracy = acc(
                data[[dataset]][[S]][[dim]]$SupportVectorMachine[[kernel]][[C]]
              )
            )
          )
        }
      }
    }
  }
}
  
# Save this object so we don't have to run that loop again
save(D, file = "Figures/dataObj.rda")

# Load the data
load("Figures/dataObj.rda")

baselineAcc <- list(
  DecisionTree = data.frame(
    dataset = character(0),
    accuracy = numeric(0)
  ),
  kNearestNeighbours = data.frame(
    dataset = character(0),
    p = numeric(0),
    accuracy = numeric(0)
  ),
  NaiveBayes = data.frame(
    dataset = character(0),
    accuracy = numeric(0)
  ),
  RandomForest = data.frame(
    dataset = character(0),
    n_estimators = numeric(0),
    accuracy = numeric(0)
  ),
  SupportVectorMachine = data.frame(
    dataset = character(0),
    kernel = character(0),
    accuracy = numeric(0)
  )
)

for (dataset in names(datasets)) {
  baselineAcc$DecisionTree <- rbind(
    baselineAcc$DecisionTree,
    data.frame(
      dataset = dataset,
      accuracy = max(
        D$DecisionTree$accuracy[
          (D$DecisionTree$dataset==dataset) & (D$DecisionTree$strategy=="No")
        ]
      )
    )
  )
  
  for (p in unique(D$kNearestNeighbours$p)) {
    baselineAcc$kNearestNeighbours <- rbind(
      baselineAcc$kNearestNeighbours,
      data.frame(
        dataset = dataset,
        p = p,
        accuracy = max(
          D$kNearestNeighbours$accuracy[
            (D$kNearestNeighbours$dataset==dataset) &
            (D$kNearestNeighbours$strategy=="No") &
            (D$kNearestNeighbours$p == p)
          ]
        )
      )
    )
  }
  
  baselineAcc$NaiveBayes <- rbind(
    baselineAcc$NaiveBayes,
    data.frame(
      dataset = dataset,
      accuracy = D$NaiveBayes$accuracy[
        (D$NaiveBayes$dataset == dataset) & (D$NaiveBayes$strategy == "No")
      ]
    )
  )
  
  for (n_estimators in unique(D$RandomForest$n_estimators)) {
    baselineAcc$RandomForest <- rbind(
      baselineAcc$RandomForest,
      data.frame(
        dataset = dataset,
        n_estimators = n_estimators,
        accuracy = max(
          D$RandomForest$accuracy[
            (D$RandomForest$dataset==dataset) &
            (D$RandomForest$strategy=="No") &
            (D$RandomForest$n_estimators == n_estimators)
          ]
        )
      )
    )
  }
  
  for (kernel in unique(D$SupportVectorMachine$kernel)) {
    baselineAcc$SupportVectorMachine <- rbind(
      baselineAcc$SupportVectorMachine,
      data.frame(
        dataset = dataset,
        kernel = kernel,
        accuracy = max(
          D$SupportVectorMachine$accuracy[
            (D$SupportVectorMachine$dataset==dataset) &
            (D$SupportVectorMachine$strategy=="No") &
            (D$SupportVectorMachine$kernel == kernel)
          ]
        )
      )
    )
  }
}

##### DECISION TREE FIGURES #####

# Strategy Summary Bar Graphs
DT_fig <- function(plotData, title) {
  P <- ggplot(data = plotData) +
    geom_col(
      mapping = aes(
        x = factor(dataset, levels = names(datasets)),
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
    scale_x_discrete(
      name = "Dataset"
    ) +
    theme(axis.text.x = element_text(angle = 30, hjust = 1)) +
    scale_y_continuous(
      name = "Accuracy",
      breaks = 0:5 / 5,
      limits = c(0,1),
      expand = c(0,0)
    ) + ggtitle(title)
  
  return(P)
}

if (!dir.exists("Figures/DecisionTree")) {
  dir.create("Figures/DecisionTree")
}

for (i in 1:length(strategies)) {
  for (
    dim in unique(
      D$DecisionTree$dim[D$DecisionTree$strategy == strategies[i]]
    )
  ) {
    P <- DT_fig(
      D$DecisionTree[
        (D$DecisionTree$strategy==strategies[i]) & (D$DecisionTree$dim==dim),
      ],
      strategy_titles[[i]](dim)
    )
    if (dim != "") {
      filename <- paste0(toString(i), "_", dim, ".png")
    } else {
      filename <- paste0(toString(i), ".png")
    }
    ggsave(
      paste0("Figures/DecisionTree/", filename),
      width = 2000,
      height = 1000,
      units = "px"
    )
  }
}

# Aggregation Grid Plot
ggplot(data = D$DecisionTree[D$DecisionTree$strategy == "Aggregate",]) +
  geom_col(
    mapping = aes(x = dim, y = accuracy, fill = as.character(depth)),
    position = position_dodge(),
    color = "black"
  ) +
  geom_line(
    data = rbind(baselineAcc$DecisionTree, baselineAcc$DecisionTree),
    mapping = aes(x=rep(c(-Inf, Inf), each = length(datasets)), y=accuracy)
  ) +
  facet_wrap(
    ~dataset,
    nrow = 3,
    ncol = 3,
    scales = "free"
  ) +
  plotTheme +
  scale_x_discrete(
    name = "Group",
    labels = c("C", "F", "P")
  ) +
  scale_y_continuous(
    name = "Accuracy",
    breaks = 0:5 / 5,
    expand = c(0,0),
    limits = c(0,1)
  ) +
  scale_fill_discrete(
    name = "Depth",
    type = c("white", "grey", "black")
  )

ggsave(
  "Figures/DecisionTree/Aggregate_grid.png",
  width = 2500,
  height = 2500,
  units = "px"
)

# PCA Grid Plot
ggplot(data = D$DecisionTree[D$DecisionTree$strategy == "PCA",]) +
  geom_rect(
    data = baselineAcc$DecisionTree,
    mapping = aes(xmin = -Inf, xmax = Inf, ymin = 0, ymax = accuracy),
    fill = "grey"
  ) +
  geom_line(
    mapping = aes(x=strtoi(dim), y=accuracy, linetype=as.character(depth))
  ) +
  facet_wrap(
    ~dataset,
    nrow = 3,
    ncol = 3,
    scales = "free"
  ) +
  plotTheme +
  scale_x_continuous(
    name = "Principle Components",
    breaks = 2:9,
    expand = c(0,0),
    limits = c(2,9)
  ) +
  scale_y_continuous(
    name = "Accuracy",
    breaks = 0:5 / 5,
    expand = c(0,0),
    limits = c(0,1)
  ) +
  scale_linetype_discrete(
    name = "Depth"
  )

ggsave(
  "Figures/DecisionTree/PCA_grid.png",
  width = 2500,
  height = 2500,
  units = "px"
)

# Kernelized PCA Grid Plot
ggplot(data = D$DecisionTree[D$DecisionTree$strategy == "Kernelized PCA",]) +
  geom_rect(
    data = baselineAcc$DecisionTree,
    mapping = aes(xmin = -Inf, xmax = Inf, ymin = 0, ymax = accuracy),
    fill = "grey"
  ) +
  geom_line(
    mapping = aes(x=strtoi(dim), y=accuracy, linetype=as.character(depth))
  ) +
  facet_wrap(
    ~dataset,
    nrow = 3,
    ncol = 3,
    scales = "free"
  ) +
  plotTheme +
  scale_x_continuous(
    name = "Principle Components",
    breaks = 2:9,
    expand = c(0,0),
    limits = c(2,9)
  ) +
  scale_y_continuous(
    name = "Accuracy",
    breaks = 0:5 / 5,
    expand = c(0,0),
    limits = c(0,1)
  ) +
  scale_linetype_discrete(
    name = "Depth"
  )

ggsave(
  "Figures/DecisionTree/KernelizedPCA_grid.png",
  width = 2500,
  height = 2500,
  units = "px"
)

# NMF Grid Plot
ggplot(data = D$DecisionTree[D$DecisionTree$strategy == "NMF",]) +
  geom_rect(
    data = baselineAcc$DecisionTree,
    mapping = aes(xmin = -Inf, xmax = Inf, ymin = 0, ymax = accuracy),
    fill = "grey"
  ) +
  geom_line(
    mapping = aes(x=strtoi(dim), y=accuracy, linetype=as.character(depth))
  ) +
  facet_wrap(
    ~dataset,
    nrow = 3,
    ncol = 3,
    scales = "free"
  ) +
  plotTheme +
  scale_x_continuous(
    name = "Features",
    breaks = 2:9,
    expand = c(0,0),
    limits = c(2,9)
  ) +
  scale_y_continuous(
    name = "Accuracy",
    breaks = 0:5 / 5,
    expand = c(0,0),
    limits = c(0,1)
  ) +
  scale_linetype_discrete(
    name = "Depth"
  )

ggsave(
  "Figures/DecisionTree/NMF_grid.png",
  width = 2500,
  height = 2500,
  units = "px"
)

##### K-NEAREST NEIGHBOURS FIGURES #####

kNN_fig <- function(plotData, title) {
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
      scales = "free_x"
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
    ggtitle(title)
    
  return(P)
}

if (!dir.exists("Figures/kNearestNeighbours")) {
  dir.create("Figures/kNearestNeighbours")
}

for (i in 1:length(strategies)) {
  for (
    dim in unique(
      D$kNearestNeighbours$dim[D$kNearestNeighbours$strategy == strategies[i]]
    )
  ) {
    P <- kNN_fig(
      D$kNearestNeighbours[
        (D$kNearestNeighbours$strategy==strategies[i]) &
        (D$kNearestNeighbours$dim==dim),
      ],
      strategy_titles[[i]](dim)
    )
    if (dim != "") {
      filename <- paste0(toString(i), "_", dim, ".png")
    } else {
      filename <- paste0(toString(i), ".png")
    }
    ggsave(
      paste0("Figures/kNearestNeighbours/", filename),
      width = 2000,
      height = 1000,
      units = "px"
    )
  }
}

for (p in unique(D$kNearestNeighbours$p)) {
  # Aggregation Grid Plot
  ggplot(
    data=D$kNearestNeighbours[
      (D$kNearestNeighbours$strategy == "Aggregate") &
      (D$kNearestNeighbours$p == p),
    ]
  ) +
    geom_col(
      mapping = aes(x = dim, y = accuracy, fill = as.character(k)),
      position = position_dodge(),
      color = "black"
    ) +
    geom_line(
      data = rbind(
        baselineAcc$kNearestNeighbours[baselineAcc$kNearestNeighbours$p==p,],
        baselineAcc$kNearestNeighbours[baselineAcc$kNearestNeighbours$p==p,]
      ),
      mapping = aes(x=rep(c(-Inf, Inf), each = length(datasets)), y=accuracy)
    ) +
    facet_wrap(
      ~dataset,
      nrow = 3,
      ncol = 3,
      scales = "free"
    ) +
    plotTheme +
    scale_x_discrete(
      name = "Group",
      labels = c("C", "F", "P")
    ) +
    scale_y_continuous(
      name = "Accuracy",
      breaks = 0:5 / 5,
      expand = c(0,0),
      limits = c(0,1)
    ) +
    scale_fill_discrete(
      name = "k",
      type = c("white", "grey", "black")
    ) +
    ggtitle(paste0("p = ", toString(p)))
  
  ggsave(
    paste0("Figures/kNearestNeighbours/Aggregate",toString(p),"_grid.png"),
    width = 2500,
    height = 2600,
    units = "px"
  )
  
  # PCA Grid Plot
  ggplot(
    data = D$kNearestNeighbours[
      (D$kNearestNeighbours$strategy == "PCA") & (D$kNearestNeighbours$p == p),
    ]
  ) +
    geom_rect(
      data=baselineAcc$kNearestNeighbours[baselineAcc$kNearestNeighbours$p==p,],
      mapping = aes(xmin = -Inf, xmax = Inf, ymin = 0, ymax = accuracy),
      fill = "grey"
    ) +
    geom_line(
      mapping = aes(x=strtoi(dim), y=accuracy, linetype=as.character(k))
    ) +
    facet_wrap(
      ~dataset,
      nrow = 3,
      ncol = 3,
      scales = "free"
    ) +
    plotTheme +
    scale_x_continuous(
      name = "Principle Components",
      breaks = 2:9,
      expand = c(0,0),
      limits = c(2,9)
    ) +
    scale_y_continuous(
      name = "Accuracy",
      breaks = 0:5 / 5,
      expand = c(0,0),
      limits = c(0,1)
    ) +
    scale_linetype_discrete(
      name = "k"
    ) + ggtitle(paste0("p = ", toString(p)))
  
  ggsave(
    paste0("Figures/kNearestNeighbours/PCA",toString(p),"_grid.png"),
    width = 2500,
    height = 2600,
    units = "px"
  )
  
  # Kernelized PCA Grid Plot
  ggplot(
    data = D$kNearestNeighbours[
      (D$kNearestNeighbours$strategy == "Kernelized PCA") &
      (D$kNearestNeighbours$p == p),
    ]
  ) +
    geom_rect(
      data=baselineAcc$kNearestNeighbours[baselineAcc$kNearestNeighbours$p==p,],
      mapping = aes(xmin = -Inf, xmax = Inf, ymin = 0, ymax = accuracy),
      fill = "grey"
    ) +
    geom_line(
      mapping = aes(x=strtoi(dim), y=accuracy, linetype=as.character(k))
    ) +
    facet_wrap(
      ~dataset,
      nrow = 3,
      ncol = 3,
      scales = "free"
    ) +
    plotTheme +
    scale_x_continuous(
      name = "Principle Components",
      breaks = 2:9,
      expand = c(0,0),
      limits = c(2,9)
    ) +
    scale_y_continuous(
      name = "Accuracy",
      breaks = 0:5 / 5,
      expand = c(0,0),
      limits = c(0,1)
    ) +
    scale_linetype_discrete(
      name = "k"
    ) + ggtitle(paste0("p = ", toString(p)))
  
  ggsave(
    paste0("Figures/kNearestNeighbours/KernelizedPCA",toString(p),"_grid.png"),
    width = 2500,
    height = 2600,
    units = "px"
  )
  
  # NMF Grid Plot
  ggplot(
    data = D$kNearestNeighbours[
      (D$kNearestNeighbours$strategy == "NMF") & (D$kNearestNeighbours$p == p),
    ]
  ) +
    geom_rect(
      data=baselineAcc$kNearestNeighbours[baselineAcc$kNearestNeighbours$p==p,],
      mapping = aes(xmin = -Inf, xmax = Inf, ymin = 0, ymax = accuracy),
      fill = "grey"
    ) +
    geom_line(
      mapping = aes(x=strtoi(dim), y=accuracy, linetype=as.character(k))
    ) +
    facet_wrap(
      ~dataset,
      nrow = 3,
      ncol = 3,
      scales = "free"
    ) +
    plotTheme +
    scale_x_continuous(
      name = "Features",
      breaks = 2:9,
      expand = c(0,0),
      limits = c(2,9)
    ) +
    scale_y_continuous(
      name = "Accuracy",
      breaks = 0:5 / 5,
      expand = c(0,0),
      limits = c(0,1)
    ) +
    scale_linetype_discrete(
      name = "k"
    ) + ggtitle(paste0("p = ", toString(p)))
  
  ggsave(
    paste0("Figures/kNearestNeighbours/NMF",toString(p),"_grid.png"),
    width = 2500,
    height = 2600,
    units = "px"
  )
}

##### NAIVE BAYES FIGURES #####

NB_fig <- function(plotData, title) {
  P <- ggplot(data = plotData) +
    geom_col(
      mapping = aes(
        x = dataset,
        y = accuracy,
      ),
      fill = "white",
      color = "black"
    ) +
    plotTheme +
    scale_x_discrete(
      name = "Dataset"
    ) +
    theme(axis.text.x = element_text(angle = 30, hjust = 1)) +
    scale_y_continuous(
      name = "Accuracy",
      breaks = 0:5 / 5,
      limits = c(0,1),
      expand = c(0,0)
    ) + ggtitle(title)
  
  return(P)
}

if (!dir.exists("Figures/NaiveBayes")) {
  dir.create("Figures/NaiveBayes")
}

for (i in 1:length(strategies)) {
  for (
    dim in unique(
      D$NaiveBayes$dim[D$NaiveBayes$strategy == strategies[i]]
    )
  ) {
    P <- NB_fig(
      D$NaiveBayes[
        (D$NaiveBayes$strategy==strategies[i]) & (D$NaiveBayes$dim==dim),
      ],
      strategy_titles[[i]](dim)
    )
    if (dim != "") {
      filename <- paste0(toString(i), "_", dim, ".png")
    } else {
      filename <- paste0(toString(i), ".png")
    }
    ggsave(
      paste0("Figures/NaiveBayes/", filename),
      width = 2000,
      height = 1000,
      units = "px"
    )
  }
}

# Aggregation Grid Plot
ggplot(data = D$NaiveBayes[D$NaiveBayes$strategy == "Aggregate",]) +
  geom_col(
    mapping = aes(x = dim, y = accuracy),
    position = position_dodge(),
    color = "black",
    fill = "white"
  ) +
  geom_line(
    data = rbind(baselineAcc$NaiveBayes, baselineAcc$NaiveBayes),
    mapping = aes(x=rep(c(-Inf, Inf), each = length(datasets)), y=accuracy)
  ) +
  facet_wrap(
    ~dataset,
    nrow = 3,
    ncol = 3,
    scales = "free"
  ) +
  plotTheme +
  scale_x_discrete(
    name = "Group",
    labels = c("C", "F", "P")
  ) +
  scale_y_continuous(
    name = "Accuracy",
    breaks = 0:5 / 5,
    expand = c(0,0),
    limits = c(0,1)
  )

ggsave(
  "Figures/NaiveBayes/Aggregate_grid.png",
  width = 2500,
  height = 2500,
  units = "px"
)

# PCA Grid Plot
ggplot(data = D$NaiveBayes[D$NaiveBayes$strategy == "PCA",]) +
  geom_rect(
    data = baselineAcc$NaiveBayes,
    mapping = aes(xmin = -Inf, xmax = Inf, ymin = 0, ymax = accuracy),
    fill = "grey"
  ) +
  geom_line(
    mapping = aes(x=strtoi(dim), y=accuracy)
  ) +
  facet_wrap(
    ~dataset,
    nrow = 3,
    ncol = 3,
    scales = "free"
  ) +
  plotTheme +
  scale_x_continuous(
    name = "Principle Components",
    breaks = 2:9,
    expand = c(0,0),
    limits = c(2,9)
  ) +
  scale_y_continuous(
    name = "Accuracy",
    breaks = 0:5 / 5,
    expand = c(0,0),
    limits = c(0,1)
  )

ggsave(
  "Figures/NaiveBayes/PCA_grid.png",
  width = 2500,
  height = 2500,
  units = "px"
)

# Kernelized PCA Grid Plot
ggplot(data = D$NaiveBayes[D$NaiveBayes$strategy == "Kernelized PCA",]) +
  geom_rect(
    data = baselineAcc$NaiveBayes,
    mapping = aes(xmin = -Inf, xmax = Inf, ymin = 0, ymax = accuracy),
    fill = "grey"
  ) +
  geom_line(
    mapping = aes(x=strtoi(dim), y=accuracy)
  ) +
  facet_wrap(
    ~dataset,
    nrow = 3,
    ncol = 3,
    scales = "free"
  ) +
  plotTheme +
  scale_x_continuous(
    name = "Principle Components",
    breaks = 2:9,
    expand = c(0,0),
    limits = c(2,9)
  ) +
  scale_y_continuous(
    name = "Accuracy",
    breaks = 0:5 / 5,
    expand = c(0,0),
    limits = c(0,1)
  )

ggsave(
  "Figures/NaiveBayes/KernelizedPCA_grid.png",
  width = 2500,
  height = 2500,
  units = "px"
)

# NMF Grid Plot
ggplot(data = D$NaiveBayes[D$NaiveBayes$strategy == "NMF",]) +
  geom_rect(
    data = baselineAcc$NaiveBayes,
    mapping = aes(xmin = -Inf, xmax = Inf, ymin = 0, ymax = accuracy),
    fill = "grey"
  ) +
  geom_line(
    mapping = aes(x=strtoi(dim), y=accuracy)
  ) +
  facet_wrap(
    ~dataset,
    nrow = 3,
    ncol = 3,
    scales = "free"
  ) +
  plotTheme +
  scale_x_continuous(
    name = "Features",
    breaks = 2:9,
    expand = c(0,0),
    limits = c(2,9)
  ) +
  scale_y_continuous(
    name = "Accuracy",
    breaks = 0:5 / 5,
    expand = c(0,0),
    limits = c(0,1)
  )

ggsave(
  "Figures/NaiveBayes/NMF_grid.png",
  width = 2500,
  height = 2500,
  units = "px"
)

##### RANDOM FOREST FIGURES #####

RF_fig <- function(plotData, title) {
  P <- ggplot(data = plotData) +
    geom_col(
      mapping = aes(
        x = factor(n_estimators, levels = c(100,200,500)),
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
      scales = "free_x"
    ) +
    plotTheme +
    scale_fill_discrete(
      name = "Depth",
      type = c("white", "grey", "black")
    ) +
    scale_x_discrete(
      name = "Number of Estimators"
    ) +
    scale_y_continuous(
      name = "Accuracy",
      breaks = 0:5 / 5,
      limits = c(0,1),
      expand = c(0,0)
    ) +
    ggtitle(title) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  return(P)
}

if (!dir.exists("Figures/RandomForest")) {
  dir.create("Figures/RandomForest")
}

for (i in 1:length(strategies)) {
  for (
    dim in unique(
      D$RandomForest$dim[D$RandomForest$strategy == strategies[i]]
    )
  ) {
    P <- RF_fig(
      D$RandomForest[
        (D$RandomForest$strategy==strategies[i]) & (D$RandomForest$dim==dim),
      ],
      strategy_titles[[i]](dim)
    )
    if (dim != "") {
      filename <- paste0(toString(i), "_", dim, ".png")
    } else {
      filename <- paste0(toString(i), ".png")
    }
    ggsave(
      paste0("Figures/RandomForest/", filename),
      width = 2000,
      height = 1000,
      units = "px"
    )
  }
}

for (n_estimators in unique(D$RandomForest$n_estimators)) {
  # Aggregation Grid Plot
  ggplot(
    data=D$RandomForest[
      (D$RandomForest$strategy == "Aggregate") &
      (D$RandomForest$n_estimators == n_estimators),
    ]
  ) +
    geom_col(
      mapping = aes(x = dim, y = accuracy, fill = as.character(depth)),
      position = position_dodge(),
      color = "black"
    ) +
    geom_line(
      data = rbind(
        baselineAcc$RandomForest[
          baselineAcc$RandomForest$n_estimators == n_estimators,
        ],
        baselineAcc$RandomForest[
          baselineAcc$RandomForest$n_estimators == n_estimators,
        ]
      ),
      mapping = aes(x=rep(c(-Inf, Inf), each = length(datasets)), y=accuracy)
    ) +
    facet_wrap(
      ~dataset,
      nrow = 3,
      ncol = 3,
      scales = "free"
    ) +
    plotTheme +
    scale_x_discrete(
      name = "Group",
      labels = c("C", "F", "P")
    ) +
    scale_y_continuous(
      name = "Accuracy",
      breaks = 0:5 / 5,
      expand = c(0,0),
      limits = c(0,1)
    ) +
    scale_fill_discrete(
      name = "Depth",
      type = c("white", "grey", "black")
    ) +
    ggtitle(paste0("Number of Estimators = ", toString(n_estimators)))
  
  ggsave(
    paste0("Figures/RandomForest/Aggregate",toString(n_estimators),"_grid.png"),
    width = 2500,
    height = 2600,
    units = "px"
  )
  
  # PCA Grid Plot
  ggplot(
    data = D$RandomForest[
      (D$RandomForest$strategy == "PCA") &
      (D$RandomForest$n_estimators == n_estimators),
    ]
  ) +
    geom_rect(
      data=baselineAcc$RandomForest[
        baselineAcc$RandomForest$n_estimators == n_estimators,
      ],
      mapping = aes(xmin = -Inf, xmax = Inf, ymin = 0, ymax = accuracy),
      fill = "grey"
    ) +
    geom_line(
      mapping = aes(x=strtoi(dim), y=accuracy, linetype=as.character(depth))
    ) +
    facet_wrap(
      ~dataset,
      nrow = 3,
      ncol = 3,
      scales = "free"
    ) +
    plotTheme +
    scale_x_continuous(
      name = "Principle Components",
      breaks = 2:9,
      expand = c(0,0),
      limits = c(2,9)
    ) +
    scale_y_continuous(
      name = "Accuracy",
      breaks = 0:5 / 5,
      expand = c(0,0),
      limits = c(0,1)
    ) +
    scale_linetype_discrete(
      name = "Depth"
    ) + ggtitle(paste0("Number of Estimators = ", toString(n_estimators)))
  
  ggsave(
    paste0("Figures/RandomForest/PCA",toString(n_estimators),"_grid.png"),
    width = 2500,
    height = 2600,
    units = "px"
  )
  
  # Kernelized PCA Grid Plot
  ggplot(
    data = D$RandomForest[
      (D$RandomForest$strategy == "Kernelized PCA") &
      (D$RandomForest$n_estimators == n_estimators),
    ]
  ) +
    geom_rect(
      data=baselineAcc$RandomForest[
        baselineAcc$RandomForest$n_estimators == n_estimators,
      ],
      mapping = aes(xmin = -Inf, xmax = Inf, ymin = 0, ymax = accuracy),
      fill = "grey"
    ) +
    geom_line(
      mapping = aes(x=strtoi(dim), y=accuracy, linetype=as.character(depth))
    ) +
    facet_wrap(
      ~dataset,
      nrow = 3,
      ncol = 3,
      scales = "free"
    ) +
    plotTheme +
    scale_x_continuous(
      name = "Principle Components",
      breaks = 2:9,
      expand = c(0,0),
      limits = c(2,9)
    ) +
    scale_y_continuous(
      name = "Accuracy",
      breaks = 0:5 / 5,
      expand = c(0,0),
      limits = c(0,1)
    ) +
    scale_linetype_discrete(
      name = "Depth"
    ) + ggtitle(paste0("Number of Estimators = ", toString(n_estimators)))
  
  ggsave(
    paste0(
      "Figures/RandomForest/KernelizedPCA",
      toString(n_estimators),
      "_grid.png"
    ),
    width = 2500,
    height = 2600,
    units = "px"
  )
  
  # NMF Grid Plot
  ggplot(
    data = D$RandomForest[
      (D$RandomForest$strategy == "NMF") &
      (D$RandomForest$n_estimators == n_estimators),
    ]
  ) +
    geom_rect(
      data=baselineAcc$RandomForest[
        baselineAcc$RandomForest$n_estimators == n_estimators,
      ],
      mapping = aes(xmin = -Inf, xmax = Inf, ymin = 0, ymax = accuracy),
      fill = "grey"
    ) +
    geom_line(
      mapping = aes(x=strtoi(dim), y=accuracy, linetype=as.character(depth))
    ) +
    facet_wrap(
      ~dataset,
      nrow = 3,
      ncol = 3,
      scales = "free"
    ) +
    plotTheme +
    scale_x_continuous(
      name = "Features",
      breaks = 2:9,
      expand = c(0,0),
      limits = c(2,9)
    ) +
    scale_y_continuous(
      name = "Accuracy",
      breaks = 0:5 / 5,
      expand = c(0,0),
      limits = c(0,1)
    ) +
    scale_linetype_discrete(
      name = "Depth"
    ) + ggtitle(paste0("Number of Estimators = ", toString(n_estimators)))
  
  ggsave(
    paste0("Figures/RandomForest/NMF",toString(n_estimators),"_grid.png"),
    width = 2500,
    height = 2600,
    units = "px"
  )
}

##### SUPPORT VECTOR MACHINE FIGURES #####

grey_scale <- c("white", "#D2D2D2", "#A8A8A8", "#7E7E7E", "#545454",
  "#2A2A2A", "black")

blue_scale <- c("lightblue", "#90B4D5", "#7390C4", "#576CB3", "#3A48A2",
                "#1D2491", "navy")

SVM_fig <- function(plotData, title) {
  P <- ggplot(data = plotData) +
    geom_col(
      mapping = aes(
        x = factor(kernel, levels = c("rbf", "linear", "quadratic", "cubic")),
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
      scales = "free_x"
    ) +
    plotTheme +
    scale_fill_discrete(
      name = "log10(C)",
      type = grey_scale,
      labels = -3:3
    ) +
    scale_x_discrete(
      name = "Kernel",
      labels = c("r", "1", "2", "3")
    ) +
    scale_y_continuous(
      name = "Accuracy",
      breaks = 0:5 / 5,
      limits = c(0,1),
      expand = c(0,0)
    ) +
    ggtitle(title)
  
  return(P)
}

if (!dir.exists("Figures/SupportVectorMachine")) {
  dir.create("Figures/SupportVectorMachine")
}

for (i in 1:length(strategies)) {
  for (
    dim in unique(
      D$SupportVectorMachine$dim[D$SupportVectorMachine$strategy==strategies[i]]
    )
  ) {
    P <- SVM_fig(
      D$SupportVectorMachine[
        (D$SupportVectorMachine$strategy==strategies[i]) &
        (D$SupportVectorMachine$dim==dim),
      ],
      strategy_titles[[i]](dim)
    )
    if (dim != "") {
      filename <- paste0(toString(i), "_", dim, ".png")
    } else {
      filename <- paste0(toString(i), ".png")
    }
    ggsave(
      paste0("Figures/SupportVectorMachine/", filename),
      width = 2000,
      height = 1000,
      units = "px"
    )
  }
}

for (kernel in unique(D$SupportVectorMachine$kernel)) {
  # Aggregation Grid Plot
  ggplot(
    data=D$SupportVectorMachine[
      (D$SupportVectorMachine$strategy == "Aggregate") &
      (D$SupportVectorMachine$kernel == kernel),
    ]
  ) +
    geom_col(
      mapping = aes(x=dim, y=accuracy, fill=factor(C,levels=sort(unique(C)))),
      position = position_dodge(),
      color = "black"
    ) +
    geom_line(
      data = rbind(
        baselineAcc$SupportVectorMachine[
          baselineAcc$SupportVectorMachine$kernel == kernel,
        ],
        baselineAcc$SupportVectorMachine[
          baselineAcc$SupportVectorMachine$kernel == kernel,
        ]
      ),
      mapping = aes(x=rep(c(-Inf, Inf), each = length(datasets)), y=accuracy)
    ) +
    facet_wrap(
      ~dataset,
      nrow = 3,
      ncol = 3,
      scales = "free"
    ) +
    plotTheme +
    scale_x_discrete(
      name = "Group",
      labels = c("C", "F", "P")
    ) +
    scale_y_continuous(
      name = "Accuracy",
      breaks = 0:5 / 5,
      expand = c(0,0),
      limits = c(0,1)
    ) +
    scale_fill_discrete(
      name = "log10(C)",
      labels = -3:3,
      type = grey_scale
    ) +
    ggtitle(paste0("Kernel: ", kernel))
  
  ggsave(
    paste0("Figures/SupportVectorMachine/Aggregate",kernel,"_grid.png"),
    width = 2500,
    height = 2600,
    units = "px"
  )
  
  # PCA Grid Plot
  ggplot(
    data = D$SupportVectorMachine[
      (D$SupportVectorMachine$strategy == "PCA") &
      (D$SupportVectorMachine$kernel == kernel),
    ]
  ) +
    geom_rect(
      data=baselineAcc$SupportVectorMachine[
        baselineAcc$SupportVectorMachine$kernel==kernel,
      ],
      mapping = aes(xmin = -Inf, xmax = Inf, ymin = 0, ymax = accuracy),
      fill = "grey"
    ) +
    geom_line(
      mapping = aes(
        x = strtoi(dim),
        y = accuracy,
        color = factor(C,levels=sort(unique(C)))
      )
    ) +
    facet_wrap(
      ~dataset,
      nrow = 3,
      ncol = 3,
      scales = "free"
    ) +
    plotTheme +
    scale_x_continuous(
      name = "Principle Components",
      breaks = 2:9,
      expand = c(0,0),
      limits = c(2,9)
    ) +
    scale_y_continuous(
      name = "Accuracy",
      breaks = 0:5 / 5,
      expand = c(0,0),
      limits = c(0,1)
    ) +
    scale_color_discrete(
      name = "log10(C)",
      type = blue_scale,
      labels = -3:3
    ) + ggtitle(paste0("Kernel: ", kernel))
  
  ggsave(
    paste0("Figures/SupportVectorMachine/PCA",kernel,"_grid.png"),
    width = 2500,
    height = 2600,
    units = "px"
  )
  
  # Kernelized PCA Grid Plot
  ggplot(
    data = D$SupportVectorMachine[
      (D$SupportVectorMachine$strategy == "Kernelized PCA") &
      (D$SupportVectorMachine$kernel == kernel),
    ]
  ) +
    geom_rect(
      data=baselineAcc$SupportVectorMachine[
        baselineAcc$SupportVectorMachine$kernel==kernel,
      ],
      mapping = aes(xmin = -Inf, xmax = Inf, ymin = 0, ymax = accuracy),
      fill = "grey"
    ) +
    geom_line(
      mapping = aes(
        x = strtoi(dim),
        y = accuracy,
        color = factor(C,levels=sort(unique(C)))
      )
    ) +
    facet_wrap(
      ~dataset,
      nrow = 3,
      ncol = 3,
      scales = "free"
    ) +
    plotTheme +
    scale_x_continuous(
      name = "Principle Components",
      breaks = 2:9,
      expand = c(0,0),
      limits = c(2,9)
    ) +
    scale_y_continuous(
      name = "Accuracy",
      breaks = 0:5 / 5,
      expand = c(0,0),
      limits = c(0,1)
    ) +
    scale_color_discrete(
      name = "log10(C)",
      type = blue_scale,
      labels = -3:3
    ) + ggtitle(paste0("Kernel: ", kernel))
  
  ggsave(
    paste0("Figures/SupportVectorMachine/KernelizedPCA",kernel,"_grid.png"),
    width = 2500,
    height = 2600,
    units = "px"
  )
  
  # NMF Grid Plot
  ggplot(
    data = D$SupportVectorMachine[
      (D$SupportVectorMachine$strategy == "NMF") &
      (D$SupportVectorMachine$kernel == kernel),
    ]
  ) +
    geom_rect(
      data=baselineAcc$SupportVectorMachine[
        baselineAcc$SupportVectorMachine$kernel==kernel,
      ],
      mapping = aes(xmin = -Inf, xmax = Inf, ymin = 0, ymax = accuracy),
      fill = "grey"
    ) +
    geom_line(
      mapping = aes(
        x = strtoi(dim),
        y = accuracy,
        color = factor(C,levels=sort(unique(C)))
      )
    ) +
    facet_wrap(
      ~dataset,
      nrow = 3,
      ncol = 3,
      scales = "free"
    ) +
    plotTheme +
    scale_x_continuous(
      name = "Features",
      breaks = 2:9,
      expand = c(0,0),
      limits = c(2,9)
    ) +
    scale_y_continuous(
      name = "Accuracy",
      breaks = 0:5 / 5,
      expand = c(0,0),
      limits = c(0,1)
    ) +
    scale_color_discrete(
      name = "log10(C)",
      type = blue_scale,
      labels = -3:3
    ) + ggtitle(paste0("Kernel: ", kernel))
  
  ggsave(
    paste0("Figures/SupportVectorMachine/NMF",kernel,"_grid.png"),
    width = 2500,
    height = 2600,
    units = "px"
  )
}

##### COARSE-GRAIN PLOTS #####

coarse_D <- data.frame(
  dataset = character(0),
  strategy = character(0),
  model = character(0),
  accuracy = numeric(0)
)

for (model in names(D)) {
  for (dataset in names(datasets)) {
    for (strategy in strategies) {
      coarse_D <- rbind(
        coarse_D,
        data.frame(
          dataset = dataset,
          strategy = strategy,
          model = model,
          accuracy = max(
            D[[model]]$accuracy[
              (D[[model]]$dataset == dataset) &
              (D[[model]]$strategy == strategy)
            ]
          )
        )
      )
    }
  }
}

coarse_D$strategy[coarse_D$strategy == "Kernelized PCA"] <- "kPCA"
coarse_D$strategy[coarse_D$strategy == "Aggregate"] <- "Aggr"
coarse_D$model[coarse_D$model == "DecisionTree"] <- "DT"
coarse_D$model[coarse_D$model == "kNearestNeighbours"] <- "k-NN"
coarse_D$model[coarse_D$model == "NaiveBayes"] <- "NB"
coarse_D$model[coarse_D$model == "RandomForest"] <- "RF"
coarse_D$model[coarse_D$model == "SupportVectorMachine"] <- "SVM"

for (model in unique(coarse_D$model)) {
  ggplot(
    data = coarse_D[(coarse_D$model == model) & (coarse_D$strategy != "No"),]
  ) +
    facet_wrap(
      . ~ dataset,
      scales = "free",
      nrow = 3,
      ncol = 3
    ) +
    geom_col(
      mapping = aes(
        x = factor(strategy, levels = c("Aggr", "PCA", "kPCA", "NMF")),
        y = accuracy
      ),
      position = position_dodge2(padding = 0),
      color = "black",
      fill = "white"
    ) +
    geom_line(
      data = rbind(
        coarse_D[(coarse_D$model == model) & (coarse_D$strategy == "No"),],
        coarse_D[(coarse_D$model == model) & (coarse_D$strategy == "No"),]
      ),
      mapping = aes(
        x = rep(c(-Inf, Inf), each = length(strategy)/2),
        y = accuracy
      )
    ) +
    plotTheme +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    scale_y_continuous(
      name = "Optimal Accuracy",
      breaks = 0:5 / 5,
      expand = c(0,0),
      limits = c(0,1)
    ) +
    scale_x_discrete(name = "Dimensionality Reduction Strategy") +
    guides(fill = guide_none()) +
    ggtitle(model)
  
  ggsave(
    paste0("Figures/coarseGrainedSummary_", model, ".png"),
    width = 2500,
    height = 2600,
    units = "px"
  )
}

##### DIFFERENCE FROM BASELINE #####

diff_D <- data.frame(
  dataset = character(0),
  strategy = character(0),
  model = character(0),
  diff = numeric(0)
)

for (dataset in unique(coarse_D$dataset)) {
  for (model in unique(coarse_D$model)) {
    for (strategy in unique(coarse_D$strategy)) {
      if (strategy != "No") {
        diff_D <- rbind(
          diff_D,
          data.frame(
            dataset = dataset,
            strategy = strategy,
            model = model,
            diff = coarse_D$accuracy[
              (coarse_D$dataset == dataset) &
              (coarse_D$strategy == strategy) &
              (coarse_D$model == model)
            ] - coarse_D$accuracy[
              (coarse_D$dataset == dataset) &
              (coarse_D$strategy == "No") &
              (coarse_D$model == model)
            ]
          )
        )
      }
    }
  }
}

diff_D$strategy <- factor(
  diff_D$strategy,
  levels = c("Aggr", "PCA", "kPCA", "NMF")
)

diff_D$dataset <- factor(
  diff_D$dataset,
  levels = names(datasets)[length(datasets):1]
)

ggplot(data = diff_D) +
  geom_rect(
    mapping = aes(
      xmin = stage(strategy, after_scale = xmin - 0.5),
      xmax = stage(strategy, after_scale = xmax + 0.5),
      ymin = stage(dataset, after_scale = ymin - 0.5),
      ymax = stage(dataset, after_scale = ymax + 0.5),
      fill = diff
    ),
    color = "black"
  ) +
  facet_grid(
    . ~ model,
    scale = "free_x",
    space = "free_x"
  ) +
  scale_x_discrete(
    name = "Dimensionality Reduction Strategy",
    expand = c(0.25,0)
  ) +
  scale_y_discrete(
    name = "Dataset",
    expand = c(0.075,0)
  ) +
  plotTheme +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  ) +
  scale_fill_gradient2(
    name = "Difference\nfrom\nBaseline",
    low = "red",
    high = "blue",
    mid = "white",
    limits = c(-0.18, 0.18),
    breaks = c(-0.15, 0, 0.15)
  )

ggsave(
  "Figures/DiffHeatMap.png",
  width = 2500,
  height = 1250,
  units = "px"
)

##### BASELINE SUMMARY #####

ggplot(data = coarse_D[coarse_D$strategy == "No",]) +
  geom_col(
    mapping = aes(
      x = factor(dataset, levels = names(datasets)),
      y = accuracy,
      fill = model
    ),
    position = position_dodge(),
    color = "black"
  ) +
  plotTheme +
  scale_x_discrete(name = "Dataset", expand = c(0,0)) +
  scale_y_continuous(
    name = "Optimal Accuracy",
    limits = c(0,1),
    breaks = 0:5/5,
    expand = c(0,0)
  ) +
  scale_fill_discrete(
    name = "Model",
    type = c("white", "grey", "black", "red", "blue")
  ) +
  ggtitle("No Dimensionality Reduction")

ggsave(
  "Figures/BaselineSummary.png",
  width = 2000,
  height = 1000,
  units = "px"
)

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

##### SAVE DATA #####

for (model in models) {
  write.table(
    D[[model]],
    file = paste0("Figures/", model, "/DataTable.tsv"),
    sep = "\t",
    row.names = FALSE,
    quote = FALSE
  )
}
