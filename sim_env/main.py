from sim_env.env import EVChargingEnv, EnvConfig

def main():
    config = EnvConfig(
        start_time=0,
        end_time=3600,
        time_step=60,
    )

    env = EVChargingEnv(config=config)
    metrics = env.run()

    print(metrics)


if __name__ == "__main__":
    main()