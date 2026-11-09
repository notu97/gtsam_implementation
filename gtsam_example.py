from importlib.metadata import version; print(version('gtsam'))

import gtsam
from gtsam.symbol_shorthand import X

graph = gtsam.NonlinearFactorGraph()
prior_noise = gtsam.noiseModel.Diagonal.Sigmas([0.3, 0.3, 0.1])
odom_noise = gtsam.noiseModel.Diagonal.Sigmas([0.2, 0.2, 0.1])

graph.add(gtsam.PriorFactorPose2(X(0), gtsam.Pose2(), prior_noise))
graph.add(gtsam.BetweenFactorPose2(
    X(0), X(1), gtsam.Pose2(2.0, 0.0, 0.0), odom_noise
))

initial = gtsam.Values()
initial.insert(X(0), gtsam.Pose2(0.2, -0.1, 0.05))
initial.insert(X(1), gtsam.Pose2(2.3, 0.2, -0.05))

result = gtsam.LevenbergMarquardtOptimizer(graph, initial).optimize()
print(result.atPose2(X(1)))