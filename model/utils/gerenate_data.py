import random
import numpy as np
import matplotlib.pyplot as plt

def get_data_synethtic(data_cluster_locations=[0, 45, 80, 100, 135, 180, -135, -100, -80, -45], 
                       num_data_points_per_cluster=200, sigma=0.05, dimension=2,
                       seed=42):

    random.seed(seed)
    np.random.seed(seed)

    cluster_centers_positive_start = np.array([data_cluster_locations[5], data_cluster_locations[0]]) *np.pi/180
    cluster_centers_negative_start = np.array([data_cluster_locations[5], data_cluster_locations[0]]) *np.pi/180
    cluster_centers_positive_nonTerminal = np.array([data_cluster_locations[4], data_cluster_locations[1]]) *np.pi/180
    cluster_centers_negative_nonTerminal = np.array([data_cluster_locations[6], data_cluster_locations[9]]) *np.pi/180
    cluster_centers_positive_terminal = np.array([data_cluster_locations[3], data_cluster_locations[2]]) *np.pi/180
    cluster_centers_negative_terminal = np.array([data_cluster_locations[7], data_cluster_locations[8]]) *np.pi/180

    data_x, data_y, data_path = [], [], []
    for angle_idx in range(len(cluster_centers_positive_terminal)):
        x_0 = gen_data_on_hypersphere(dim=dimension, phi_angles=np.random.randn(num_data_points_per_cluster)*sigma + cluster_centers_positive_start[angle_idx])
        x_1 = gen_data_on_hypersphere(dim=dimension, phi_angles=np.random.randn(num_data_points_per_cluster)*sigma + cluster_centers_positive_nonTerminal[angle_idx])
        x_2 = gen_data_on_hypersphere(dim=dimension, phi_angles=np.random.randn(num_data_points_per_cluster)*sigma + cluster_centers_positive_terminal[angle_idx])
        data_x += list(np.stack((x_0, x_1, x_2), axis=1))
        data_y += [np.array([[1],[1],[1]])  for _ in range(num_data_points_per_cluster)]
        data_path += [np.array([[angle_idx],[angle_idx],[angle_idx]])  for _ in range(num_data_points_per_cluster)]

    for angle_idx in range(len(cluster_centers_negative_terminal)):
        x_0 = gen_data_on_hypersphere(dim=dimension, phi_angles=np.random.randn(num_data_points_per_cluster)*sigma + cluster_centers_negative_start[angle_idx])
        x_1 = gen_data_on_hypersphere(dim=dimension, phi_angles=np.random.randn(num_data_points_per_cluster)*sigma + cluster_centers_negative_nonTerminal[angle_idx])
        x_2 = gen_data_on_hypersphere(dim=dimension, phi_angles=np.random.randn(num_data_points_per_cluster)*sigma + cluster_centers_negative_terminal[angle_idx])
        data_x += list(np.stack((x_0, x_1, x_2), axis=1))
        data_y += [np.array([[0],[0],[0]])  for _ in range(num_data_points_per_cluster)]
        data_path += [np.array([[angle_idx],[angle_idx],[angle_idx]])  for _ in range(num_data_points_per_cluster)]

    temp = list(zip(data_x, data_y, data_path))
    random.shuffle(temp)
    data_x, data_y, data_path = zip(*temp)
    data_x, data_y, data_path = list(data_x), list(data_y), list(data_path)

    return {"data_x": data_x, "data_y": data_y, "data_path": data_path}


def gen_data_on_hypersphere(dim, phi_angles, theta_angles=None):
    dataset = np.zeros((len(phi_angles), dim))
    if dim==2:
        dataset[:, 0] = np.cos(phi_angles)
        dataset[:, 1] = np.sin(phi_angles)
        return dataset
    elif dim==3:
        dataset[:, 0] = np.sin(theta_angles) * np.cos(phi_angles)
        dataset[:, 1] = np.sin(theta_angles) * np.sin(phi_angles)
        dataset[:, 2] = np.cos(theta_angles)
        return dataset
    else:
        raise ValueError("Dimension should be 2 or 3!")


def get_data_synethtic_downstream_test(data_cluster_locations=[0, 45, 80, 100, 135, 180, -135, -100, -80, -45], 
                                       num_data_points_per_cluster=100, sigma=0.05, dimension=2,
                                       seed=42):

    random.seed(seed)
    np.random.seed(seed)

    cluster_centers_positive_start = np.array([data_cluster_locations[5], data_cluster_locations[0]]) *np.pi/180
    cluster_centers_negative_start = np.array([data_cluster_locations[5], data_cluster_locations[0]]) *np.pi/180
    cluster_centers_positive_nonTerminal = np.array([data_cluster_locations[4], data_cluster_locations[1]]) *np.pi/180
    cluster_centers_negative_nonTerminal = np.array([data_cluster_locations[6], data_cluster_locations[9]]) *np.pi/180
    cluster_centers_positive_terminal = np.array([data_cluster_locations[3], data_cluster_locations[2]]) *np.pi/180
    cluster_centers_negative_terminal = np.array([data_cluster_locations[7], data_cluster_locations[8]]) *np.pi/180

    data_x, data_y, data_path, data_cluster = [], [], [], []
    cluster_states_positive = [np.array([[5],[4],[3]]), np.array([[0],[1],[2]])]
    cluster_states_negative = [np.array([[5],[6],[7]]), np.array([[0],[9],[8]])]

    for angle_idx in range(len(cluster_centers_positive_terminal)):
        x_0 = gen_data_on_hypersphere(dim=dimension, phi_angles=np.random.randn(num_data_points_per_cluster)*sigma + cluster_centers_positive_start[angle_idx])
        x_1 = gen_data_on_hypersphere(dim=dimension, phi_angles=np.random.randn(num_data_points_per_cluster)*sigma + cluster_centers_positive_nonTerminal[angle_idx])
        x_2 = gen_data_on_hypersphere(dim=dimension, phi_angles=np.random.randn(num_data_points_per_cluster)*sigma + cluster_centers_positive_terminal[angle_idx])
        data_x += list(np.stack((x_0, x_1, x_2), axis=1))
        data_y += [np.array([[1],[1],[1]])  for _ in range(num_data_points_per_cluster)]
        data_path += [np.array([[angle_idx],[angle_idx],[angle_idx]])  for _ in range(num_data_points_per_cluster)]
        data_cluster += [cluster_states_positive[angle_idx]  for _ in range(num_data_points_per_cluster)]
    
    for angle_idx in range(len(cluster_centers_negative_terminal)):
        x_0 = gen_data_on_hypersphere(dim=dimension, phi_angles=np.random.randn(num_data_points_per_cluster)*sigma + cluster_centers_negative_start[angle_idx])
        x_1 = gen_data_on_hypersphere(dim=dimension, phi_angles=np.random.randn(num_data_points_per_cluster)*sigma + cluster_centers_negative_nonTerminal[angle_idx])
        x_2 = gen_data_on_hypersphere(dim=dimension, phi_angles=np.random.randn(num_data_points_per_cluster)*sigma + cluster_centers_negative_terminal[angle_idx])
        data_x += list(np.stack((x_0, x_1, x_2), axis=1))
        data_y += [np.array([[0],[0],[0]])  for _ in range(num_data_points_per_cluster)]
        data_path += [np.array([[angle_idx],[angle_idx],[angle_idx]])  for _ in range(num_data_points_per_cluster)]
        data_cluster += [cluster_states_negative[angle_idx]  for _ in range(num_data_points_per_cluster)]

    temp = list(zip(data_x, data_y, data_path, data_cluster))
    random.shuffle(temp)
    data_x, data_y, data_path, data_cluster = zip(*temp)

    data_x, data_y, data_path, data_cluster = list(data_x), list(data_y), list(data_path), list(data_cluster)

    return {"data_x": data_x, "data_y": data_y, "data_path": data_path, "data_cluster": data_cluster}


def plot_hypersphere(data_xx, data_yy, proportion_visualized=0.1, title_txt="", data_pp=None, marker_size=500, fig_size=(7, 7), elev=None, azim=None):
    data_x = np.vstack(data_xx[-int(len(data_xx)*proportion_visualized):])
    data_y = data_yy[-int(len(data_yy)*proportion_visualized):]
    data_o = np.vstack([[0, 1, 2] for _ in data_y]).flatten()
    data_y = np.vstack(data_y)
    if data_pp is not None: 
        data_path = np.vstack(data_pp[-int(len(data_pp)*proportion_visualized):])

    if data_x.shape[1]==2:
        plot_hypersphere_2D(data_x, data_y, data_o, title_txt, data_path, marker_size, fig_size)
    elif data_x.shape[1]==3:
        plot_hypersphere_3D(data_x, data_y, data_o, title_txt, data_path, marker_size, fig_size, elev, azim)


def plot_hypersphere_2D(data_x, data_y, data_o, title_txt="", data_path=None, marker_size=500, fig_size=(7, 7)):
    # Plotting
    unit_hypersphere = plt.Circle((0, 0), 1, color='k', alpha=0.1)
    fig, ax = plt.subplots()
    fig.set_size_inches(fig_size[0], fig_size[1])
    ax.add_patch(unit_hypersphere)
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    if data_path is None:
        lo_risk_terminal = data_x[np.where((data_y.flatten()==0) & (data_o==2))[0]]
        hi_risk_terminal = data_x[np.where((data_y.flatten()==1) & (data_o==2))[0]]

        lo_risk_non_terminal = data_x[np.where((data_y.flatten()==0) & (data_o==1))[0]]
        hi_risk_non_terminal = data_x[np.where((data_y.flatten()==1) & (data_o==1))[0]]

        lo_risk_start = data_x[np.where((data_y.flatten()==0) & (data_o==0))[0]]
        hi_risk_start = data_x[np.where((data_y.flatten()==1) & (data_o==0))[0]]
        
        plt.scatter(lo_risk_terminal[:, 0], lo_risk_terminal[:, 1], color='blue', s=100, label='Low Risk - Terminal')
        plt.scatter(hi_risk_terminal[:, 0], hi_risk_terminal[:, 1], color='red', s=100, label='High Risk - Terminal')
        plt.scatter(lo_risk_non_terminal[:, 0], lo_risk_non_terminal[:, 1], color='royalblue', s=100, label='Low Risk - Non-Terminal')
        plt.scatter(hi_risk_non_terminal[:, 0], hi_risk_non_terminal[:, 1], color='indianred', s=100, label='High Risk - Non-Terminal')
        plt.scatter(lo_risk_start[:, 0], lo_risk_start[:, 1], color='darkblue', s=100, label='Low Risk - Terminal')
        plt.scatter(hi_risk_start[:, 0], hi_risk_start[:, 1], color='darkred', s=100, label='High Risk - Terminal')
    else:
        lo_risk_terminal_0 = data_x[np.where((data_y.flatten()==0) & (data_o==2) & (data_path.flatten()==0))[0]]
        lo_risk_terminal_1 = data_x[np.where((data_y.flatten()==0) & (data_o==2) & (data_path.flatten()==1))[0]]
        hi_risk_terminal_0 = data_x[np.where((data_y.flatten()==1) & (data_o==2) & (data_path.flatten()==0))[0]]
        hi_risk_terminal_1 = data_x[np.where((data_y.flatten()==1) & (data_o==2) & (data_path.flatten()==1))[0]]

        lo_risk_non_terminal_0 = data_x[np.where((data_y.flatten()==0) & (data_o==1) & (data_path.flatten()==0))[0]]
        lo_risk_non_terminal_1 = data_x[np.where((data_y.flatten()==0) & (data_o==1) & (data_path.flatten()==1))[0]]
        hi_risk_non_terminal_0 = data_x[np.where((data_y.flatten()==1) & (data_o==1) & (data_path.flatten()==0))[0]]
        hi_risk_non_terminal_1 = data_x[np.where((data_y.flatten()==1) & (data_o==1) & (data_path.flatten()==1))[0]]

        lo_risk_start_0 = data_x[np.where((data_y.flatten()==0) & (data_o==0) & (data_path.flatten()==0))[0]]
        lo_risk_start_1 = data_x[np.where((data_y.flatten()==0) & (data_o==0) & (data_path.flatten()==1))[0]]
        hi_risk_start_0 = data_x[np.where((data_y.flatten()==1) & (data_o==0) & (data_path.flatten()==0))[0]]
        hi_risk_start_1 = data_x[np.where((data_y.flatten()==1) & (data_o==0) & (data_path.flatten()==1))[0]]
        
        plt.scatter(lo_risk_terminal_0[:, 0], lo_risk_terminal_0[:, 1], color='blue', s=marker_size, label='Low Risk - Terminal', marker="*", edgecolors='black')
        plt.scatter(hi_risk_terminal_0[:, 0], hi_risk_terminal_0[:, 1], color='red', s=marker_size*0.8, label='High Risk - Terminal', marker="*", edgecolors='black')
        plt.scatter(lo_risk_terminal_1[:, 0], lo_risk_terminal_1[:, 1], color='blue', s=marker_size, label='Low Risk - Terminal', marker="P", edgecolors='black')
        plt.scatter(hi_risk_terminal_1[:, 0], hi_risk_terminal_1[:, 1], color='red', s=marker_size*0.8, label='High Risk - Terminal', marker="P", edgecolors='black')
        
        plt.scatter(lo_risk_non_terminal_0[:, 0], lo_risk_non_terminal_0[:, 1], color='royalblue', s=marker_size, label='Low Risk - Non-Terminal', marker="v", edgecolors='black')
        plt.scatter(hi_risk_non_terminal_0[:, 0], hi_risk_non_terminal_0[:, 1], color='indianred', s=marker_size*0.8, label='High Risk - Non-Terminal', marker="v", edgecolors='black')
        plt.scatter(lo_risk_non_terminal_1[:, 0], lo_risk_non_terminal_1[:, 1], color='royalblue', s=marker_size, label='Low Risk - Non-Terminal', marker="^", edgecolors='black')
        plt.scatter(hi_risk_non_terminal_1[:, 0], hi_risk_non_terminal_1[:, 1], color='indianred', s=marker_size*0.8, label='High Risk - Non-Terminal', marker="^", edgecolors='black')
        
        plt.scatter(lo_risk_start_0[:, 0], lo_risk_start_0[:, 1], color='lightskyblue', s=marker_size, label='Low Risk - Terminal', marker="o", edgecolors='black')
        plt.scatter(hi_risk_start_0[:, 0], hi_risk_start_0[:, 1], color='orange', s=marker_size*0.8, label='High Risk - Terminal', marker="o", edgecolors='black')
        plt.scatter(lo_risk_start_1[:, 0], lo_risk_start_1[:, 1], color='lightskyblue', s=marker_size, label='Low Risk - Terminal', marker="s", edgecolors='black')
        plt.scatter(hi_risk_start_1[:, 0], hi_risk_start_1[:, 1], color='orange', s=marker_size*0.8, label='High Risk - Terminal', marker="s", edgecolors='black')

    # plt.legend()
    plt.title(title_txt)
    plt.show()


def plot_hypersphere_3D(data_x, data_y, data_o, title_txt="", data_path=None, marker_size=500, fig_size=(7, 7), elev=None, azim=None):

    # Plotting
    fig = plt.figure()
    fig.set_size_inches(fig_size[0], fig_size[1])
    ax = fig.add_subplot(projection='3d')

    r = 0.99
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
    x = r*(np.cos(u) * np.sin(v))
    y = r*(np.sin(u) * np.sin(v))
    z = r*(np.cos(v))
    ax.plot_surface(x, y, z, cmap=plt.cm.YlGnBu_r, alpha=0.2)

    if data_path is None:
        lo_risk_terminal = data_x[np.where((data_y.flatten()==0) & (data_o==2))[0]]
        hi_risk_terminal = data_x[np.where((data_y.flatten()==1) & (data_o==2))[0]]

        lo_risk_non_terminal = data_x[np.where((data_y.flatten()==0) & (data_o==1))[0]]
        hi_risk_non_terminal = data_x[np.where((data_y.flatten()==1) & (data_o==1))[0]]

        lo_risk_start = data_x[np.where((data_y.flatten()==0) & (data_o==0))[0]]
        hi_risk_start = data_x[np.where((data_y.flatten()==1) & (data_o==0))[0]]
        
        ax.scatter(lo_risk_terminal[:, 0], lo_risk_terminal[:, 1], lo_risk_terminal[:, 2], color='blue', s=100, label='Low Risk - Terminal')
        ax.scatter(hi_risk_terminal[:, 0], hi_risk_terminal[:, 1], hi_risk_terminal[:, 2], color='red', s=100, label='High Risk - Terminal')
        ax.scatter(lo_risk_non_terminal[:, 0], lo_risk_non_terminal[:, 1], lo_risk_non_terminal[:, 2], color='royalblue', s=100, label='Low Risk - Non-Terminal')
        ax.scatter(hi_risk_non_terminal[:, 0], hi_risk_non_terminal[:, 1], hi_risk_non_terminal[:, 2], color='indianred', s=100, label='High Risk - Non-Terminal')
        ax.scatter(lo_risk_start[:, 0], lo_risk_start[:, 1], lo_risk_start[:, 2], color='darkblue', s=100, label='Low Risk - Terminal')
        ax.scatter(hi_risk_start[:, 0], hi_risk_start[:, 1], hi_risk_start[:, 2], color='darkred', s=100, label='High Risk - Terminal')
    else:
        lo_risk_terminal_0 = data_x[np.where((data_y.flatten()==0) & (data_o==2) & (data_path.flatten()==0))[0]]
        lo_risk_terminal_1 = data_x[np.where((data_y.flatten()==0) & (data_o==2) & (data_path.flatten()==1))[0]]
        hi_risk_terminal_0 = data_x[np.where((data_y.flatten()==1) & (data_o==2) & (data_path.flatten()==0))[0]]
        hi_risk_terminal_1 = data_x[np.where((data_y.flatten()==1) & (data_o==2) & (data_path.flatten()==1))[0]]

        lo_risk_non_terminal_0 = data_x[np.where((data_y.flatten()==0) & (data_o==1) & (data_path.flatten()==0))[0]]
        lo_risk_non_terminal_1 = data_x[np.where((data_y.flatten()==0) & (data_o==1) & (data_path.flatten()==1))[0]]
        hi_risk_non_terminal_0 = data_x[np.where((data_y.flatten()==1) & (data_o==1) & (data_path.flatten()==0))[0]]
        hi_risk_non_terminal_1 = data_x[np.where((data_y.flatten()==1) & (data_o==1) & (data_path.flatten()==1))[0]]

        lo_risk_start_0 = data_x[np.where((data_y.flatten()==0) & (data_o==0) & (data_path.flatten()==0))[0]]
        lo_risk_start_1 = data_x[np.where((data_y.flatten()==0) & (data_o==0) & (data_path.flatten()==1))[0]]
        hi_risk_start_0 = data_x[np.where((data_y.flatten()==1) & (data_o==0) & (data_path.flatten()==0))[0]]
        hi_risk_start_1 = data_x[np.where((data_y.flatten()==1) & (data_o==0) & (data_path.flatten()==1))[0]]
        
        ax.scatter(lo_risk_terminal_0[:, 0], lo_risk_terminal_0[:, 1], lo_risk_terminal_0[:, 2], 
                   color='blue', s=marker_size, label='Low Risk - Terminal', marker="*", edgecolors='black')
        ax.scatter(hi_risk_terminal_0[:, 0], hi_risk_terminal_0[:, 1], hi_risk_terminal_0[:, 2], 
                   color='red', s=marker_size*0.8, label='High Risk - Terminal', marker="*", edgecolors='black')
        ax.scatter(lo_risk_terminal_1[:, 0], lo_risk_terminal_1[:, 1], lo_risk_terminal_1[:, 2], 
                   color='blue', s=marker_size, label='Low Risk - Terminal', marker="P", edgecolors='black')
        ax.scatter(hi_risk_terminal_1[:, 0], hi_risk_terminal_1[:, 1], hi_risk_terminal_1[:, 2], 
                   color='red', s=marker_size*0.8, label='High Risk - Terminal', marker="P", edgecolors='black')
        
        ax.scatter(lo_risk_non_terminal_0[:, 0], lo_risk_non_terminal_0[:, 1], lo_risk_non_terminal_0[:, 2], 
                   color='royalblue', s=marker_size, label='Low Risk - Non-Terminal', marker="v", edgecolors='black')
        ax.scatter(hi_risk_non_terminal_0[:, 0], hi_risk_non_terminal_0[:, 1], hi_risk_non_terminal_0[:, 2], 
                   color='indianred', s=marker_size*0.8, label='High Risk - Non-Terminal', marker="v", edgecolors='black')
        ax.scatter(lo_risk_non_terminal_1[:, 0], lo_risk_non_terminal_1[:, 1], lo_risk_non_terminal_1[:, 2], 
                   color='royalblue', s=marker_size, label='Low Risk - Non-Terminal', marker="^", edgecolors='black')
        ax.scatter(hi_risk_non_terminal_1[:, 0], hi_risk_non_terminal_1[:, 1], hi_risk_non_terminal_1[:, 2], 
                   color='indianred', s=marker_size*0.8, label='High Risk - Non-Terminal', marker="^", edgecolors='black')
        
        ax.scatter(lo_risk_start_0[:, 0], lo_risk_start_0[:, 1], lo_risk_start_0[:, 2], 
                   color='lightskyblue', s=marker_size, label='Low Risk - Terminal', marker="o", edgecolors='black')
        ax.scatter(hi_risk_start_0[:, 0], hi_risk_start_0[:, 1], hi_risk_start_0[:, 2], 
                   color='orangered', s=marker_size*0.8, label='High Risk - Terminal', marker="o", edgecolors='black')
        ax.scatter(lo_risk_start_1[:, 0], lo_risk_start_1[:, 1], lo_risk_start_1[:, 2], 
                   color='lightskyblue', s=marker_size, label='Low Risk - Terminal', marker="s", edgecolors='black')
        ax.scatter(hi_risk_start_1[:, 0], hi_risk_start_1[:, 1], hi_risk_start_1[:, 2], 
                   color='orangered', s=marker_size*0.8, label='High Risk - Terminal', marker="s", edgecolors='black')
    
    
    # ax.plot_surface(x, y, z, cmap='gist_gray', alpha=0.2)
    if elev is not None and azim is not None:
        ax.view_init(elev, azim)

    # Turn off x, y, and z tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    max_range = np.array([1.2, 1.1, 1.1]).max() / 1.4

    mid_x = (0) * 0.5
    mid_y = (0) * 0.5
    mid_z = (0) * 0.5
    ax.set_xlim(mid_x - max_range-0.1, mid_x + max_range+0.1)
    ax.set_ylim(mid_y - max_range-0.1, mid_y + max_range+0.1)
    ax.set_zlim(mid_z - max_range+0.2, mid_z + max_range-0.2)

    plt.title(title_txt)