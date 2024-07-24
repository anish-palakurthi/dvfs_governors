#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/cpufreq.h>
#include <linux/slab.h>
#include <linux/sched.h>
#include <linux/time.h>

#define NUM_CORES 4
#define NUM_STATES 10  // More granular frequency states
#define NUM_ACTIONS 5  // More actions: large decrease, small decrease, maintain, small increase, large increase
#define ALPHA 0.1      // Learning rate
#define GAMMA 0.9      // Discount factor
#define EPSILON 0.1    // Exploration rate

struct q_table {
    float q1[NUM_STATES][NUM_ACTIONS];
    float q2[NUM_STATES][NUM_ACTIONS];
};

static struct q_table *q_tables[NUM_CORES];
static int current_states[NUM_CORES];
static unsigned long long last_update_time[NUM_CORES];
static unsigned long long energy_consumed[NUM_CORES];

// Initialize Q-tables
static void init_q_tables(void)
{
    int i, j, k;
    for (i = 0; i < NUM_CORES; i++) {
        q_tables[i] = kmalloc(sizeof(struct q_table), GFP_KERNEL);
        if (!q_tables[i]) {
            pr_err("Failed to allocate memory for Q-table\n");
            return;
        }
        for (j = 0; j < NUM_STATES; j++) {
            for (k = 0; k < NUM_ACTIONS; k++) {
                q_tables[i]->q1[j][k] = 0.0;
                q_tables[i]->q2[j][k] = 0.0;
            }
        }
        current_states[i] = 0;
        last_update_time[i] = ktime_get_ns();
        energy_consumed[i] = 0;
    }
}

// Choose action using epsilon-greedy policy
static int choose_action(int core)
{
    if (get_random_u32() < (UINT_MAX * EPSILON)) {
        return get_random_u32() % NUM_ACTIONS;
    } else {
        int best_action = 0;
        float max_q = q_tables[core]->q1[current_states[core]][0] + q_tables[core]->q2[current_states[core]][0];
        int i;
        for (i = 1; i < NUM_ACTIONS; i++) {
            float q_sum = q_tables[core]->q1[current_states[core]][i] + q_tables[core]->q2[current_states[core]][i];
            if (q_sum > max_q) {
                max_q = q_sum;
                best_action = i;
            }
        }
        return best_action;
    }
}

// Update Q-tables
static void update_q_tables(int core, int action, int next_state, float reward)
{
    float max_future_q1 = q_tables[core]->q1[next_state][0];
    float max_future_q2 = q_tables[core]->q2[next_state][0];
    int i;
    for (i = 1; i < NUM_ACTIONS; i++) {
        if (q_tables[core]->q1[next_state][i] > max_future_q1) {
            max_future_q1 = q_tables[core]->q1[next_state][i];
        }
        if (q_tables[core]->q2[next_state][i] > max_future_q2) {
            max_future_q2 = q_tables[core]->q2[next_state][i];
        }
    }
    
    if (get_random_u32() % 2) {
        q_tables[core]->q1[current_states[core]][action] += ALPHA * (reward + GAMMA * max_future_q2 - q_tables[core]->q1[current_states[core]][action]);
    } else {
        q_tables[core]->q2[current_states[core]][action] += ALPHA * (reward + GAMMA * max_future_q1 - q_tables[core]->q2[current_states[core]][action]);
    }
    
    current_states[core] = next_state;
}

// Calculate reward based on energy consumption and performance
static float calculate_reward(int core, unsigned int cur_freq, unsigned int next_freq, unsigned long long time_diff)
{
    unsigned long long energy_diff = energy_consumed[core] - (cur_freq * time_diff / 1000000);  // Simplified energy calculation
    float performance = (float)next_freq / cur_freq;
    float energy_efficiency = performance / energy_diff;
    
    return energy_efficiency * 100 - 50;  // Normalize reward
}

// DVFS governor function
static int double_q_governor(struct cpufreq_policy *policy)
{
    unsigned int cur_freq = policy->cur;
    unsigned int next_freq;
    int action, next_state;
    float reward;
    int core = policy->cpu;
    unsigned long long current_time = ktime_get_ns();
    unsigned long long time_diff = current_time - last_update_time[core];

    // Map current frequency to state
    current_states[core] = (cur_freq - policy->min) / ((policy->max - policy->min) / NUM_STATES);

    action = choose_action(core);

    // Perform action
    switch (action) {
        case 0: // Large decrease
            next_freq = max(cur_freq - 300000, policy->min);
            break;
        case 1: // Small decrease
            next_freq = max(cur_freq - 100000, policy->min);
            break;
        case 2: // Maintain
            next_freq = cur_freq;
            break;
        case 3: // Small increase
            next_freq = min(cur_freq + 100000, policy->max);
            break;
        case 4: // Large increase
            next_freq = min(cur_freq + 300000, policy->max);
            break;
        default:
            next_freq = cur_freq;
            break;
    }

    // Calculate reward
    reward = calculate_reward(core, cur_freq, next_freq, time_diff);

    // Map next frequency to state
    next_state = (next_freq - policy->min) / ((policy->max - policy->min) / NUM_STATES);

    update_q_tables(core, action, next_state, reward);

    // Update time and energy
    last_update_time[core] = current_time;
    energy_consumed[core] += (cur_freq * time_diff / 1000000);  // Simplified energy calculation

    __cpufreq_driver_target(policy, next_freq, CPUFREQ_RELATION_L);

    return 0;
}

static struct cpufreq_governor double_q_gov = {
    .name = "double_q_governor",
    .governor = double_q_governor,
    .owner = THIS_MODULE,
};

static int __init double_q_gov_init(void)
{
    init_q_tables();
    return cpufreq_register_governor(&double_q_gov);
}

static void __exit double_q_gov_exit(void)
{
    int i;
    for (i = 0; i < NUM_CORES; i++) {
        kfree(q_tables[i]);
    }
    cpufreq_unregister_governor(&double_q_gov);
}

module_init(double_q_gov_init);
module_exit(double_q_gov_exit);

MODULE_AUTHOR("Anish Palakurthi, Jay Rountree");
MODULE_DESCRIPTION("Double-Q Learning DVFS Governor");
MODULE_LICENSE("GPL");