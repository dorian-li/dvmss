from pathlib import Path

from dvmss.agent import VehicleAgentCreator

if __name__ == "__main__":
    cessna_172_path = Path(__file__).resolve().parent / "cessna_172.stl"
    vehicle_agent_creator = VehicleAgentCreator(
        model_path=cessna_172_path,
        true_wing_length=11,
        # true_fuselage_length=8,
        longest_axis_as_wing=True,
    )
    vehicle_agent = vehicle_agent_creator.create_vehicle_agent()

    simulation = Simulation(mag_agent, geomagnetic_model, flight)
