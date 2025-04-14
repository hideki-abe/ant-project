import gymnasium as gym
import numpy as np
import os
from gymnasium.wrappers import RecordVideo
import platform

# Configurar MUJOCO_GL somente para Windows
if platform.system() == "Windows":
    os.environ["MUJOCO_GL"] = "glfw"  # GLFW geralmente é suportado no Windows


# FITNESS
def evaluate_ant_with_sine_waves(params, render=False, record_video=False, video_folder="videos",
                                 video_name="ant_video"):
    """
    Avalia o ambiente Ant usando ondas senoidais como controladores.

    Args:
        params (list): Lista com 24 parâmetros:
            - 8 primeiros parâmetros: Frequências das ondas senoidais
            - Próximos 8 parâmetros: Amplitudes das ondas senoidais (0 a 1)
            - Últimos 8 parâmetros: Deslocamentos de fase das ondas senoidais (0 a 2π)
        render (bool): Se deve renderizar o ambiente
        record_video (bool): Se deve gravar um vídeo
        video_folder (str): Pasta onde salvar o vídeo
        video_name (str): Nome do arquivo de vídeo

    Returns:
        float: Recompensa total acumulada
    """
    # Validar parâmetros
    if len(params) != 24:
        raise ValueError("São esperados 24 parâmetros (8 frequências, 8 amplitudes e 8 deslocamentos de fase)")

    # Extrair frequências, amplitudes e deslocamentos de fase
    frequencies = params[:8]
    amplitudes = params[8:16]
    phase_shifts = params[16:]

    # Garantir que as amplitudes estão entre 0 e 1
    amplitudes = np.clip(amplitudes, 0, 1)

    # Criar ambiente
    env = gym.make("Ant-v5", render_mode="rgb_array" if render or record_video else None,
                   terminate_when_unhealthy=False)

    # Envolver ambiente com gravador de vídeo, se necessário
    if record_video:
        env = RecordVideo(env, video_folder, name_prefix=video_name, episode_trigger=lambda x: 1)

    # Reiniciar ambiente
    observation, _ = env.reset()

    total_reward = 0
    timesteps = 1000

    # Simulação
    for t in range(timesteps):
        # Gerar ações usando ondas senoidais com deslocamentos de fase
        time_factor = t / 20.0  # Escalar tempo para frequências razoáveis
        actions = np.array([
            amplitudes[i] * np.sin(frequencies[i] * time_factor + phase_shifts[i])
            for i in range(8)
        ])

        # Tomar ação no ambiente
        observation, reward, terminated, truncated, info = env.step(actions)
        total_reward += reward

        if terminated or truncated:
            break

    env.close()
    return total_reward