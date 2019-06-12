model=model/pretrain_linear_reward.ckpt

mkdir -p exec_traces/normal_agent
mkdir -p exec_traces/normal_agent_constant_throughput
mkdir -p exec_traces/inclusion_agent_1_features
mkdir -p exec_traces/inclusion_agent_2_features
mkdir -p exec_traces/inclusion_agent_3_features
mkdir -p exec_traces/inclusion_agent_4_features
mkdir -p explanations
mkdir -p figures

# Step 1
#
# Run the agent on network traces and collect the execution traces
# This reflects the various states of the system during a run of Pensieve

# Main agent (for normal traces and constant throughput)
python3 run_agent.py $model net_traces/traces/* exec_traces/normal_agent/
python3 run_agent.py $model net_traces/synthetic_traces/* exec_traces/normal_agent_constant_throughput/

# Agent including only top K features
python3 run_agent_top_k.py 1 $model net_traces/traces/* exec_traces/inclusion_agent_1_features/
python3 run_agent_top_k.py 2 $model net_traces/traces/* exec_traces/inclusion_agent_2_features/
python3 run_agent_top_k.py 3 $model net_traces/traces/* exec_traces/inclusion_agent_3_features/
python3 run_agent_top_k.py 4 $model net_traces/traces/* exec_traces/inclusion_agent_4_features/


# Step 2
#
# Use LIME to generate explanations
# The 'Explanation' structure is defined in `explainer.py` and allow easy
# storage and manipulation of the traces and explanations.
#
# We generate explanations for all normal traces and constant traces that have
# a throughput close to the available bitrates.

for trace in $(ls exec_traces/normal_agent/*)
do
	echo python3 explainer.py $trace explanations/$(basename $trace .csv).explanation.pckl
done

for bitrate in 300 750 1200 1850 2850 4300
do
	echo python3 explainer.py \
		exec_traces/normal_agent_constant_throughput/constant_"$bitrate"_kbps.csv \
		constant_"$bitrate"_kbps.explanation.pckl
done
